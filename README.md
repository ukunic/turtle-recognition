# 🐢 Turtle Face Recognition System ("Kim Bu?")

## 📌 Overview
This project is an AI-powered system that autonomously detects, crops, and identifies turtle faces from photos by comparing them against a database[cite: 1]. The system is built on a "Multi-Agent" architecture rather than a traditional, unidirectional flow[cite: 1].

## 🚀 Technologies Used
*   **Core Language:** Python (3.11)[cite: 1].
*   **AI & Agents:** CrewAI & LangChain, Groq Cloud (Llama-3.3-70b-versatile)[cite: 1].
*   **Computer Vision:** OpenCV (cv2) for cropping, CLAHE lighting equalization, and SIFT/FLANN algorithms[cite: 1].
*   **Environment & UI:** Python-dotenv for key management, Streamlit for the web interface[cite: 1].

## 🤖 Multi-Agent Architecture
The system operates with two specialized AI agents working sequentially like a production line[cite: 1]:
1.  **Vision Expert:** Analyzes the raw photo, autonomously triggers the `analyze_image_tool`, and performs a Center Crop on the turtle's face based on reference dimensions[cite: 1].
2.  **Identity Matcher:** Waits for the first agent, receives the cropped image path, and triggers the `compare_database_tool` to scan the database using SIFT data, ultimately providing a detailed report in Turkish[cite: 1].

## 📈 Algorithm Evolution
The project underwent three major phases to maximize accuracy[cite: 1]:
*   **Phase 1 (Mock Data):** Established agent communication using random mock matching scores[cite: 1].
*   **Phase 2 (ORB Algorithm):** Attempted pixel comparison using OpenCV's ORB and Brute-Force Matcher, which resulted in a low accuracy rate (41.8%) due to scaling and rotation vulnerabilities[cite: 1].
*   **Phase 3 (SIFT + CLAHE + FLANN):** Integrated CLAHE for lighting equalization, SIFT for scale/rotation invariance, FLANN-based KNN for matching speed, and Lowe's Ratio Test to filter noise, achieving highly accurate biometric recognition[cite: 1].

## 🛠️ Software Engineering Principles
The codebase strictly adheres to professional standards[cite: 1]:
*   **SOLID Principles:** Implemented Single Responsibility, Open/Closed architecture for tool integration, and Dependency Inversion using `@tool` decorators[cite: 1].
*   **Clean Code:** Organized with a modular folder structure (`agents/`, `tools/`, `data/`) and Separation of Concerns (delegating heavy coordinate math to pure Python functions away from the LLM)[cite: 1].

## 🛑 Challenges & Solutions
*   **API Version Conflicts:** Switched from CrewAI's default Gemini integration to LangChain + Llama 3.3 (Groq) to resolve V1Beta 404 errors[cite: 1].
*   **LLM Format Leakage:** Used Prompt Engineering and simplified "foolproof" tools to prevent the LLM from generating XML tags or hallucinating data instead of standard JSON function calling[cite: 1].
*   **Library Conflicts:** Replaced Pydantic schemas with built-in Python Type Hints and Docstrings to fix `TypeError` conflicts between CrewAI and Streamlit[cite: 1].
*   **Rate Limits:** Integrated robust `try-except` blocks into the Streamlit UI to gracefully handle Groq API rate limits and prevent silent system crashes[cite: 1].
