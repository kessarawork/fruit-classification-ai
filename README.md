# 🍎 Fruit Classification AI

## โปรเจคนี้ทำอะไร
ระบบจำแนกผลไม้ 5 ชนิด ได้แก่ Apple, Banana, Grape, Orange และ Pineapple
โดยใช้ Deep Learning พร้อมระบบ XAI (Explainable AI) 
ที่อธิบายว่า AI โฟกัสส่วนไหนของรูปก่อนตัดสินใจ

## Features
- อัปโหลดรูปผลไม้แล้วทำนายได้ทันที
- แสดง Confidence Score ของแต่ละ class
- มี Grad-CAM และ Saliency Map อธิบายผลการทำนาย

## เทคโนโลยีที่ใช้
- Python
- TensorFlow / Keras
- Streamlit
- OpenCV

## วิธีรัน
pip install streamlit tensorflow opencv-python pillow
streamlit run app.py
