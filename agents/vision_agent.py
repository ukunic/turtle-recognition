from crewai import Agent, LLM
import os
from dotenv import load_dotenv

load_dotenv()

class TurtleAgents:
    def vision_expert(self) -> Agent:
        groq_llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )
        return Agent(
            role='Senior Computer Vision Expert',
            goal='Detect and crop the turtle face from the provided image.',
            backstory='You are an expert in wildlife image processing.',
            verbose=True,
            allow_delegation=False,
            llm=groq_llm
        )

    # --- İŞTE YENİ AJANIMIZ BURADA ---
    def matcher_expert(self) -> Agent:
        groq_llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )
        return Agent(
            role='Senior Identity Matcher',
            goal='Take the cropped image path from the vision expert and find the turtle in the database.',
            backstory='You are a data scientist specialized in matching biological patterns and identifying animals.',
            verbose=True,
            allow_delegation=False, # Şimdilik işi başkasına devretmesin, kendi yapsın
            llm=groq_llm
        )