# Importing required libraries for creating a chatbot
import nltk  # Natural Language Toolkit for text processing
import spacy  # SpaCy for advanced NLP tasks
from nltk.tokenize import word_tokenize  # Tokenizer from NLTK
from nltk.corpus import stopwords  # Stopwords list from NLTK
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF Vectorizer from scikit-learn
from sklearn.metrics.pairwise import cosine_similarity  # Cosine similarity from scikit-learn

# Downloading NLTK resources if not already done
nltk.download('punkt')  # Tokenizer models for NLTK
nltk.download('stopwords')  # Stopwords list for NLTK

# Utilization and loading of SpaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")  # Load SpaCy's small English model

# List of commonly asked FAQs with their responses
faq_pairs = [
    ("Tell me about the universe.", "The universe is all of space and time, including all forms of matter and energy. It began with the Big Bang approximately 13.8 billion years ago."),
    ("What is computer science?", "Computer science is the study of computers and computational systems. It involves both theoretical studies of algorithms and practical aspects of implementing them."),
    ("What is coding?", "Coding is the process of writing instructions for computers using programming languages. It is a crucial part of software development and is used to create applications and systems."),
    ("What's the weather like today?", "You can check the weather by visiting your preferred weather website or using a weather app for the most accurate and up-to-date information."),
    ("What's the best way to stay healthy?", "Eating a balanced diet rich in fruits and vegetables, exercising regularly, staying hydrated, and getting sufficient sleep are all important for maintaining good health."),
    ("What are some popular travel destinations this year?", "Some popular travel destinations this year include Bali, known for its beautiful beaches and culture; Paris, famous for its romantic ambiance and landmarks; Tokyo, with its vibrant city life and technology; and New York City, known for its iconic sites and diverse culture."),
    ("How can I improve my productivity?", "To improve productivity, try setting clear and achievable goals, breaking tasks into smaller steps, using productivity tools and techniques like the Pomodoro Technique, and ensuring you take regular breaks to stay refreshed."),
    ("What are some good books to read?", "Highly recommended books include 'Atomic Habits' by James Clear, which offers insights into building good habits; 'Sapiens' by Yuval Noah Harari, which explores the history of humanity; and 'The Alchemist' by Paulo Coelho, a novel about following your dreams."),
    ("How do I start learning a new language?", "To start learning a new language, you can use language-learning apps such as Duolingo or Babbel, enroll in online courses, join language exchange communities, or practice speaking with native speakers to build proficiency."),
    ("What's the latest news in technology?", "Stay updated on the latest tech news by following reputable technology news websites such as TechCrunch, Wired, The Verge, or engaging with tech news on social media platforms."),
    ("What are some fun hobbies to try?", "Fun hobbies to explore include painting, gardening, hiking, playing a musical instrument, cooking new recipes, or even learning a new skill such as photography or crafting."),
    ("How do I stay motivated when working from home?", "To stay motivated while working from home, establish a consistent daily routine, create a dedicated workspace, set clear work hours, and incorporate breaks and physical activity into your schedule."),
    ("What's the best way to manage stress?", "Effective stress management techniques include practicing mindfulness and meditation, engaging in regular physical exercise, maintaining a healthy work-life balance, and seeking support from friends, family, or a professional counselor."),
    ("What is artificial intelligence?", "Artificial intelligence (AI) is a branch of computer science focused on creating systems capable of performing tasks that typically require human intelligence, such as understanding natural language, recognizing patterns, and making decisions."),
    ("How can I save money effectively?", "To save money effectively, create a budget, track your expenses, set savings goals, and consider automating your savings by setting up regular transfers to a savings account."),
    ("What are some tips for a successful job interview?", "To succeed in a job interview, research the company and the role, practice common interview questions, dress appropriately, and be prepared to discuss your experiences and skills confidently."),
    ("How do I start a blog?", "Starting a blog involves choosing a blogging platform (like WordPress or Medium), picking a niche or topic, creating quality content, and promoting your blog through social media and SEO techniques."),
    ("What are some ways to improve my public speaking skills?", "To improve public speaking skills, practice regularly, focus on your body language, engage with your audience, and consider joining a public speaking group or taking a course for constructive feedback."),
    ("What is blockchain technology?", "Blockchain technology is a decentralized digital ledger used to record transactions across multiple computers in a way that ensures security and transparency, commonly associated with cryptocurrencies like Bitcoin."),
    ("How can I improve my time management skills?", "Improving time management skills involves setting clear priorities, creating a schedule or to-do list, avoiding procrastination, and using tools like calendars or productivity apps to stay organized."),
    ("What are some effective ways to learn coding?", "Effective ways to learn coding include taking online courses, working on projects, practicing coding exercises, joining coding communities, and using resources like coding bootcamps or tutorials."),
    ("How do I develop good study habits?", "Develop good study habits by creating a dedicated study space, setting specific goals, breaking study sessions into manageable chunks, using active learning techniques, and reviewing material regularly."),
    ("What are some eco-friendly practices I can adopt?", "Eco-friendly practices include reducing waste, recycling, using energy-efficient appliances, conserving water, supporting sustainable products, and minimizing single-use plastics."),
    ("How is the day?", "The day is going well, thank you for asking! How can I assist you today?")
]

# Preprocessing function to clean and normalize text
def preprocess_text(text):
    # Convert text to lowercase and process it using SpaCy
    doc = nlp(text.lower())
    # Lemmatize tokens and remove stopwords and non-alphabetic tokens
    tokens = [token.lemma_ for token in doc if token.text not in stopwords.words('english') and token.is_alpha]
    return ' '.join(tokens)

# Preprocess FAQ questions
processed_faqs = [(preprocess_text(question), answer) for question, answer in faq_pairs]

# Initialize TF-IDF Vectorizer to convert text data into numerical vectors
vectorizer = TfidfVectorizer()
# Extract FAQ questions for vectorization
faq_questions = [faq[0] for faq in processed_faqs]
# Create TF-IDF matrix for the FAQ questions
tfidf_matrix = vectorizer.fit_transform(faq_questions)

# Function to find the most similar FAQ based on user query
def get_most_similar_question(user_query):
    # Preprocess user query
    user_query_processed = preprocess_text(user_query)
    # Transform user query into TF-IDF vector
    user_query_vec = vectorizer.transform([user_query_processed])
    # Compute cosine similarity between user query and FAQ questions
    cosine_similarities = cosine_similarity(user_query_vec, tfidf_matrix).flatten()
    # Find index of the most similar FAQ
    best_match_idx = cosine_similarities.argmax()
    # Return the most similar FAQ answer and its confidence score
    return faq_pairs[best_match_idx][1], cosine_similarities[best_match_idx]

# Interaction of the chatbot
def faq_chatbot():
    print("Zuha's Chatbot:Welcome to the FAQ chatbot! Ask me anything about our products, services, or various topics.")
    while True:
        # Get user input and convert it to lowercase
        user_input = input("\nYou: ").strip().lower()
        
        # Handle greetings
        if user_input in ["hi", "hey", "hello"]:
            print("Zuha's Chatbot: Hello! How can I assist you today?")
            continue
        
        # Handle question about how the chatbot is doing
        if user_input in ["how are you?"]:
            print("Zuha's Chatbot: I'm doing great, thanks for asking. How about you? How can I assist you today?")
            continue
        
        # Exit condition to break the loop
        if user_input in ["exit", "quit", "bye"]:
            print("Zuha's Chatbot: Goodbye!")
            break
        
        # Get the most similar FAQ response
        response, confidence = get_most_similar_question(user_input)
        
        # Print response if confidence is above a threshold
        if confidence > 0.2:  # Adjust the threshold if needed for better performance
            print(f"Chatbot: {response}")
        else:
            print("Zuha's Chatbot: I'm sorry, I don't understand your question. Could you please rephrase it?")

# Run the chatbot if this script is executed
if __name__ == "__main__":
    faq_chatbot()


