from crewai import Task, Crew, Process
from agents.vision_agent import TurtleAgents
from tools.image_cropper import crop_image_tool
from tools.face_matcher import face_match_tool # Yeni aracımızı dahil ettik

def main():
    print("--- Kaplumbağa Yüz Tanıma Sistemi Başlatılıyor ---")
    
    # 1. Ajanlarımızı üretiyoruz
    agents_factory = TurtleAgents()
    vision_agent = agents_factory.vision_expert()
    matcher_agent = agents_factory.matcher_expert() # İkinci ajanımız geldi!
    
    # Alet çantalarını hazırlıyoruz
    vision_agent.tools = [crop_image_tool]
    matcher_agent.tools = [face_match_tool]

    # 2. Görev 1: Görüntü İşleme (Kırpma)
    crop_task = Task(
        description=(
            "We have a turtle photo located at 'data/mock_images/turtle_test.jpg'. "
            "Please use the 'crop_image_tool' to crop its face. "
            "Save the cropped result to 'data/mock_images/agent_cropped_face.jpg'."
            # (Not: Koordinatları yazmadık, Llama 3 otonom olarak bulmayı deneyecek)
        ),
        expected_output="The path to the newly cropped image file.",
        agent=vision_agent
    )

    # 3. Görev 2: Kimlik Eşleştirme (Burası Çok Önemli!)
    match_task = Task(
        description=(
            "Take the file path output from the previous cropping task. "
            "Use the 'face_match_tool' with that path to search the database. "
            "Finally, write a detailed and friendly report in TURKISH about the matched turtle."
        ),
        expected_output="A detailed Turkish report containing the matched turtle's name, ID, and match percentage.",
        agent=matcher_agent
    )

    # 4. Ekibi (Crew) Kuruyoruz
    turtle_crew = Crew(
        agents=[vision_agent, matcher_agent],
        tasks=[crop_task, match_task],
        process=Process.sequential, # İşleri SIRAYLA yap (Bant sistemi)
        verbose=True
    )

    print("\n--- Çoklu Ajanlar Göreve Başladı ---\n")
    result = turtle_crew.kickoff()
    
    print("\n\n====== NİHAİ SİSTEM RAPORU ======")
    print(result)

if __name__ == "__main__":
    main()