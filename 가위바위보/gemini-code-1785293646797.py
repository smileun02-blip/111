import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# 페이지 기본 설정
st.set_page_config(
    page_title="AI vs 인간: 가위바위보 판독기 🤖",
    page_icon="✌️",
    layout="centered"
)

# 배경색 및 유머러스한 스타일 커스텀 (CSS)
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;
    }
    .title-text {
        color: #ff4b4b;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
    }
    .sub-text {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 메인 타이틀 및 소개
st.markdown('<div class="title-text">🤖 AI의 심안: 가위바위보 판독기</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">"당신이 낼 손모양, AI는 이미 다 보고 있습니다..."</div>', unsafe_allow_html=True)

# 모델 및 라벨 로딩 (캐싱을 통해 성능 최적화)
@st.cache_resource
def load_rps_model():
    # Keras / Teachable Machine 모델 파일과 라벨 파일 불러오기
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = f.readlines()
    return model, class_names

try:
    model, class_names = load_rps_model()
except Exception as e:
    st.error(f"⚠️ 모델 또는 라벨 파일 로드 실패: {e}")
    st.info("📌 'keras_model.h5'와 'labels.txt' 파일이 파이썬 실행 파일과 같은 위치에 있는지 확인해주세요!")
    st.stop()

st.write("---")

# 안내 문구
st.info("💡 웹캠을 켜고 카메라 앞에 손을 얹은 뒤 [사진 촬영]을 눌러주세요!")

# 카메라 입력 받기
img_file_buffer = st.camera_input("AI와 눈싸움 중...")

if img_file_buffer is not None:
    # 1. 이미지 로드 및 전처리 (224x224 RGB)
    image = Image.open(img_file_buffer).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # 2. 넘파이 배열 변환 및 정규화 (티처블 머신 전처리 방식: (x / 127.5) - 1)
    image_array = np.asarray(image, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1.0
    
    # 모델 입력 형태 (1, 224, 224, 3) 배치 차원 추가
    data = np.expand_dims(normalized_image_array, axis=0)

    # 3. 모델 예측 실행
    with st.spinner("AI가 당신의 무의식을 분석하는 중... 🧠"):
        prediction = model.predict(data)
        index = np.argmax(prediction[0])
        
        # 라벨 텍스트 정제 (앞의 숫자 및 공백 제거)
        raw_class_name = class_names[index].strip()
        class_name = raw_class_name.split(' ', 1)[-1] if ' ' in raw_class_name else raw_class_name
        confidence_score = float(prediction[0][index])

    # 4. 분석 결과 출력 및 유머러스한 반응
    st.subheader("🔮 분석 완료!")
    
    # 가위, 바위, 보에 따른 유머 멘트 및 효과
    if "가위" in class_name:
        st.balloons()
        st.success(f"당신이 낸 것은 **[{class_name}]** 입니다! (확신도: {confidence_score * 100:.1f}%)")
        st.write("✂️ **AI의 한마디:** '싹둑싹둑! 혹시 제 랜선도 잘라버리시려는 건 아니죠?'")
    elif "바위" in class_name:
        st.snow()
        st.success(f"당신이 낸 것은 **[{class_name}]** 입니다! (확신도: {confidence_score * 100:.1f}%)")
        st.write("🪨 **AI의 한마디:** '묵직하네요! 주먹으로 모니터를 치시면 안 됩니다.'")
    elif "보" in class_name:
        st.balloons()
        st.success(f"당신이 낸 것은 **[{class_name}]** 입니다! (확신도: {confidence_score * 100:.1f}%)")
        st.write("🖐️ **AI의 한마디:** '하이파이브하자고요? 제 손은 화면 속에 있어서 아쉽네요!'")
    else:
        st.warning(f"인식 결과: **[{class_name}]** (확신도: {confidence_score * 100:.1f}%)")
        st.write("🤔 **AI의 한마디:** '음... 가위나 바위나 보 중 하나가 맞나요? 손가락을 조금 더 명확히 보여주세요!'")

    # 신뢰도 게이지 표시
    st.progress(confidence_score)
