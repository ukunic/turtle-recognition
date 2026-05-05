import streamlit as st
import os
from PIL import Image

from crewai import Task, Crew, Process
from crewai.tools import tool
from agents.vision_agent import TurtleAgents
from tools.face_matcher import face_match_tool

st.set_page_config(layout="wide", page_title="Kaplumbağa Tanıma Sistemi")

# --- SENIOR YAKLAŞIMI: OTOMATİK ŞEMALI ARAÇLAR ---
# args_schema'yı sildik, yerine CrewAI'nin otomatik okuyacağı tip belirteçlerini (command: str) koyduk.

@tool("analyze_image")
def analyze_image_tool(command: str) -> str:
    """Analyzes and crops the loaded image. STRICT RULE: You must provide 'start' as the command."""
    input_path = os.path.join("data", "mock_images", "ui_test_turtle.jpg")
    output_path = os.path.join("data", "mock_images", "ui_cropped_face.jpg")
    try:
        img = Image.open(input_path)
        width, height = img.size
        # Merkezi kesim (Center Crop) matematiği
        left = (width - 400) / 2
        top = (height - 400) / 2
        right = (width + 400) / 2
        bottom = (height + 400) / 2
        img.crop((left, top, right, bottom)).save(output_path)
        return "Analysis complete."
    except Exception as e:
        return f"Error: {e}"

@tool("compare_database")
def compare_database_tool(command: str) -> str:
    """Compares the analyzed image with the database. STRICT RULE: You must provide 'start' as the command."""
    output_path = os.path.join("data", "mock_images", "ui_cropped_face.jpg")
    try:
        if hasattr(face_match_tool, 'func'):
            return face_match_tool.func(image_path=output_path)
        else:
            return face_match_tool.run(image_path=output_path)
    except Exception as e:
        return f"Error: {e}"

# --- ARAYÜZ VE ÇALIŞTIRMA ---
st.title("🐢 Kaplumbağa Tanıma Sistemi")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Fotoğraf Yükle", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        input_path = os.path.join("data", "mock_images", "ui_test_turtle.jpg")
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(input_path, caption="Yüklenen Fotoğraf", use_container_width=True)

with col2:
    if uploaded_file and st.button("Analizi Başlat", use_container_width=True):
        with st.spinner("Ajan 1 analiz ediyor, Ajan 2 karşılaştırıyor..."):
            
            try:
                # 1. Ajanlar Tanımlanıyor
                agents_factory = TurtleAgents()
                vision_agent = agents_factory.vision_expert()
                matcher_agent = agents_factory.matcher_expert()
                
                # 2. Araçlar ajanlara veriliyor
                vision_agent.tools = [analyze_image_tool]
                matcher_agent.tools = [compare_database_tool]

                # 3. Saf ve net görevler
                analyze_task = Task(
                    description="Call the 'analyze_image' tool with command='start' to process the photo. Output strictly 'Analiz edildi.' and nothing else.",
                    expected_output="Confirmation that the image is analyzed.",
                    agent=vision_agent
                )

                compare_task = Task(
                    description="Call the 'compare_database' tool with command='start' to check for matches. Output the exact match percentage or result in Turkish. Do not write long reports.",
                    expected_output="The final match result in Turkish.",
                    agent=matcher_agent
                )

                # 4. Sistemi Ateşle
                crew = Crew(
                    agents=[vision_agent, matcher_agent],
                    tasks=[analyze_task, compare_task],
                    process=Process.sequential,
                    verbose=False 
                )

                result = crew.kickoff()
                
                # 5. Sonuçlar
                cropped_output_path = os.path.join("data", "mock_images", "ui_cropped_face.jpg")
                if os.path.exists(cropped_output_path):
                    st.image(cropped_output_path, caption="Analiz Edilen Merkez Bölge", width=250)
                
                st.markdown("### Sonuç")
                st.info(result.raw if hasattr(result, 'raw') else str(result))
                
            except Exception as e:
                st.error(f"🛑 HATA YAKALANDI: {str(e)}")