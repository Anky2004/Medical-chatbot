from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import uuid

# ---------- ENV / SETTINGS ----------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "your_pinecone_key"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
INDEX_NAME = "medical-chatbot"
PINECONE_REGION = "us-east-1"
DATA_DIR = "data/"

# ---------- Flask App Setup ----------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "your-secret-key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///medbot.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
scheduler = BackgroundScheduler()
scheduler.start()

# ---------- Models ----------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    medicines = db.relationship("Medicine", backref="user", lazy=True)

class Medicine(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    time = db.Column(db.String(10), nullable=False)
    frequency = db.Column(db.String(10), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

reminder_log = []

# ---------- Login Manager ----------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------- Pinecone + LangChain Setup ----------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    texts = splitter.split_documents(docs)
    pc.create_index(name=INDEX_NAME, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION))
    PineconeVectorStore.from_documents(texts, embedding=embedding, index_name=INDEX_NAME)

vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embedding)
retriever = vectorstore.as_retriever()

llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=TOGETHER_API_KEY,
    temperature=0.4,
    max_tokens=512
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Address it like a chatbot and always greet nicely. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json()
    query = data.get("message")
    if not query:
        return jsonify({"error": "No input provided"}), 400
    try:
        result = rag_chain.invoke({"input": query})
        reply = result.get("answer", "I'm not sure how to respond to that.") if isinstance(result, dict) else str(result)
        return jsonify({"reply": reply})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Something went wrong processing your request."}), 500

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        if User.query.filter_by(email=email).first():
            flash("Email already registered.")
            return redirect(url_for("signup"))
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        flash("Signup successful. Please log in.")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("medicines"))
        flash("Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/medicines", methods=["GET", "POST"])
@login_required
def medicines():
    if request.method == "POST":
        name = request.form.get("name")
        time = request.form.get("time")
        frequency = request.form.get("frequency")
        if name and time:
            med = Medicine(name=name, time=time, frequency=frequency, user_id=current_user.id)
            db.session.add(med)
            db.session.commit()

            hour, minute = map(int, time.split(":"))
            now = datetime.now()
            run_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if run_time < now:
                run_time += timedelta(days=1)

            job_id = f"{current_user.id}-{med.id}"
            scheduler.add_job(trigger_reminder, 'date', run_date=run_time,
                              args=[str(med.id), name, frequency, current_user.id], id=job_id)

        return redirect(url_for("medicines"))

    meds = Medicine.query.filter_by(user_id=current_user.id).all()
    return render_template("medicines.html", schedule=meds, reminders=reminder_log)

@app.route("/delete/<int:med_id>", methods=["POST"])
@login_required
def delete_medicine(med_id):
    med = Medicine.query.get_or_404(med_id)
    if med.user_id != current_user.id:
        flash("Unauthorized.")
        return redirect(url_for("medicines"))

    try:
        scheduler.remove_job(f"{current_user.id}-{med.id}")
    except:
        pass

    db.session.delete(med)
    db.session.commit()
    return redirect(url_for("medicines"))

@app.route("/get_reminders")
@login_required
def get_reminders():
    user_reminders = [msg for msg in reminder_log if f"user {current_user.id}" in msg]
    return jsonify({"reminders": user_reminders})

# ---------- Reminder Logic ----------
def trigger_reminder(med_id, name, frequency, user_id):
    reminder_log.append(f"⏰ {name} for user {user_id}")
    print(f"[Reminder Triggered] {name} (User: {user_id})")

    if frequency == "daily":
        next_run = datetime.now() + timedelta(days=1)
        scheduler.add_job(trigger_reminder, 'date',
            run_date=next_run.replace(hour=next_run.hour, minute=next_run.minute, second=0, microsecond=0),
            args=[med_id, name, frequency, user_id], id=f"{user_id}-{med_id}")
@app.route("/acknowledge_reminder", methods=["POST"])
@login_required
def acknowledge_reminder():
    data = request.get_json()
    med_name = data.get("name")
    reminder_log[:] = [msg for msg in reminder_log if not (med_name in msg and f"user {current_user.id}" in msg)]
    return jsonify({"status": "acknowledged"})

# ---------- Run ----------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)