import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- CONFIG ---------- #
st.set_page_config(page_title="Enneagram Deep Test", layout="wide")
st.title("🧠 Enneagram 81-Question Test (Core / Shadow / Behavior)")

# ---------- INITIAL DATA ---------- #
@st.cache_data
def load_questions():
    df = pd.read_csv("questions.csv")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

df_questions = load_questions()

# ---------- UNIVERSAL SUB-QUESTIONS ---------- #
universal_sub_questions = [
    {
        'question': 'คุณคิดว่าตัวเองตัดสินใจโดยอิงจากอะไรเป็นหลัก?',
        'choices': [
            'ความถูกต้อง / จริยธรรม (Type 1)',
            'การได้รับการรักหรือยอมรับ (Type 2)',
            'เป้าหมาย / ความสำเร็จ (Type 3)',
            'อารมณ์ภายในและเอกลักษณ์ (Type 4)',
            'ความรู้ ความเข้าใจลึก (Type 5)',
            'ความปลอดภัยและการวางแผน (Type 6)',
            'อิสระและความสนุก (Type 7)',
            'พลังอำนาจและการควบคุม (Type 8)',
            'ความสงบและหลีกเลี่ยงความขัดแย้ง (Type 9)'
        ]
    },
    {
        'question': 'คุณกลัวสิ่งใดมากที่สุดในชีวิต?',
        'choices': [
            'การผิดพลาดทางศีลธรรม (Type 1)',
            'การไม่มีใครต้องการ (Type 2)',
            'ความล้มเหลวและไม่น่าประทับใจ (Type 3)',
            'การไม่มีใครเข้าใจ (Type 4)',
            'การไม่มีข้อมูลหรือถูกควบคุม (Type 5)',
            'การไม่มีหลักยึด / โดดเดี่ยว (Type 6)',
            'ความเบื่อ / ความเจ็บปวด (Type 7)',
            'การไร้พลัง / ถูกครอบงำ (Type 8)',
            'ความขัดแย้ง / ความวุ่นวาย (Type 9)'
        ]
    }
]

# ---------- MAPPING FOR SUB-ANALYSIS ---------- #
sub_analysis_map = {
    # Q1 mapping
    'ความถูกต้อง / จริยธรรม (Type 1)':       'คุณตัดสินใจโดยคำนึงถึงหลักจริยธรรมและความถูกต้องเป็นหลัก',
    'การได้รับการรักหรือยอมรับ (Type 2)':    'คุณมักตัดสินใจโดยคำนึงถึงความสัมพันธ์และความรู้สึกของผู้อื่น',
    'เป้าหมาย / ความสำเร็จ (Type 3)':       'คุณเลือกทางที่ส่งเสริมเป้าหมายและภาพลักษณ์แห่งความสำเร็จ',
    'อารมณ์ภายในและเอกลักษณ์ (Type 4)':    'คุณให้ความสำคัญกับความรู้สึกภายในและความเป็นตัวของตัวเอง',
    'ความรู้ ความเข้าใจลึก (Type 5)':        'คุณเน้นหาข้อมูลและความเข้าใจเชิงลึกก่อนตัดสินใจ',
    'ความปลอดภัยและการวางแผน (Type 6)':   'คุณวางแผนและเตรียมพร้อมเพื่อสร้างความมั่นคง',
    'อิสระและความสนุก (Type 7)':           'คุณมองหาความสนุกและอิสระเป็นหลักในการตัดสินใจ',
    'พลังอำนาจและการควบคุม (Type 8)':      'คุณเลือกเส้นทางที่ให้คุณควบคุมสถานการณ์และเข้มแข็ง',
    'ความสงบและหลีกเลี่ยงความขัดแย้ง (Type 9)': 'คุณเลือกความสงบและหลีกเลี่ยงความขัดแย้งเป็นสำคัญ',
    # Q2 mapping
    'การผิดพลาดทางศีลธรรม (Type 1)':        'คุณกลัวทำผิดจริยธรรม จึงระมัดระวังไม่ให้ล่วงเกินมาตรฐาน',
    'การไม่มีใครต้องการ (Type 2)':           'คุณกลัวไร้คุณค่า จึงดูแลความสัมพันธ์ไว้เสมอ',
    'ความล้มเหลวและไม่น่าประทับใจ (Type 3)': 'คุณกลัวไม่สำเร็จและเสียภาพลักษณ์ จึงมุ่งมั่นสูง',
    'การไม่มีใครเข้าใจ (Type 4)':           'คุณกลัวว่าความเป็นตัวเองจะไม่ถูกเข้าใจ',
    'การไม่มีข้อมูลหรือถูกควบคุม (Type 5)': 'คุณกลัวขาดข้อมูลหรือถูกจำกัด จึงเน้นเสาะหาองค์ความรู้',
    'การไม่มีหลักยึด / โดดเดี่ยว (Type 6)': 'คุณกลัวไร้ที่พึ่ง จึงเผื่อแผนและมองหาความมั่นคง',
    'ความเบื่อ / ความเจ็บปวด (Type 7)':     'คุณกลัวความเจ็บปวดและความน่าเบื่อ จึงแสวงหาความท้าทาย',
    'การไร้พลัง / ถูกครอบงำ (Type 8)':       'คุณกลัวการถูกครอบงำ จึงยืนหยัดเพื่อพลังและอิสรภาพ',
    'ความขัดแย้ง / ความวุ่นวาย (Type 9)':   'คุณกลัวความวุ่นวาย จึงพยายามสร้างความสมดุลและสันติ'
}

# ---------- SESSION STATE ---------- #
if 'main_submitted' not in st.session_state:
    st.session_state.main_submitted = False
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'sub_submitted' not in st.session_state:
    st.session_state.sub_submitted = False
if 'sub_answers' not in st.session_state:
    st.session_state.sub_answers = {}

# ---------- MAIN FORM ---------- #
st.markdown("### โปรดให้คะแนนแต่ละข้อ (1 = ไม่ตรงเลย, 5 = ตรงมาก)")
with st.form('main_form'):
    temp_responses = []
    for idx, row in df_questions.iterrows():
        score = st.slider(
            f"ข้อ {row['Question Number']}: {row['Question (Thai)'].split('สำหรับ')[0].strip()}",
            1, 5, 3,
            key=f"main_slider_{idx}"
        )
        temp_responses.append({
            'Type': row['Enneagram Type'],
            'Category': row['Question Category'],
            'Score': score
        })
    if st.form_submit_button('ประมวลผลผลลัพธ์'):
        st.session_state.main_submitted = True
        st.session_state.responses = temp_responses

# ---------- SHOW MAIN RESULTS ---------- #
if st.session_state.main_submitted:
    df_resp = pd.DataFrame(st.session_state.responses)
    summary = df_resp.groupby(['Type','Category']).mean(numeric_only=True).reset_index()
    pivot_table = summary.pivot(index='Type', columns='Category', values='Score').fillna(0)

    st.markdown('## 📊 ผลสรุป')
    st.dataframe(pivot_table.style.background_gradient(cmap='YlGnBu'))

    core_scores = pivot_table.get('Core', pd.Series(dtype=float))
    if not core_scores.empty:
        sorted_scores = core_scores.sort_values(ascending=False)
        top_type = sorted_scores.index[0]
        second_type = sorted_scores.index[1]
        top_score = sorted_scores.iloc[0]
        second_score = sorted_scores.iloc[1]

        # Main result
        num = top_type.split(':')[0].replace('Type ','')
        label = top_type.split(': ')[1]
        st.success(f"คุณมีแนวโน้มเป็น Type {num} → {label}")

        # Always show collaborators & caution
        collaborators = sorted_scores.iloc[1:3]
        caution = sorted_scores.iloc[-2:]

        st.markdown('#### 🤝 คนที่คุณ **น่าจะทำงานด้วยได้ดี**')
        for t in collaborators.index:
            n, l = t.split(': ')
            st.markdown(f"- **{n}** ({l})")

        st.markdown('#### ⚠️ คนที่คุณ **ควรระมัดระวัง** ในการทำงานด้วย')
        for t in caution.index:
            n, l = t.split(': ')
            st.markdown(f"- **{n}** ({l})")

        # Radar Chart and sub-analysis expander
        labels = [t.replace('Type ','T') for t in core_scores.index]
        values = core_scores.tolist()
        angles = np.linspace(0,2*np.pi,len(values),endpoint=False).tolist()
        values += values[:1]; angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'polar':True})
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        st.pyplot(fig)

        if abs(top_score - second_score) < 0.2:
            st.warning('คะแนน Core ใกล้กัน โปรดตอบคำถามเสริม')
            with st.expander('🧠 คำถามเสริมเพื่อแยก Core'):
                with st.form('sub_form'):
                    for q in universal_sub_questions:
                        st.session_state.sub_answers[q['question']] = st.radio(
                            q['question'], q['choices'], key=f"sub_{q['question']}"
                        )
                    if st.form_submit_button('🔍 วิเคราะห์จากคำตอบข้างต้น'):
                        st.session_state.sub_submitted = True

                if st.session_state.sub_submitted:
                    ans1 = st.session_state.sub_answers[universal_sub_questions[0]['question']]
                    ans2 = st.session_state.sub_answers[universal_sub_questions[1]['question']]
                    text1 = sub_analysis_map.get(ans1, '')
                    text2 = sub_analysis_map.get(ans2, '')
                    st.markdown('### 💡 ผลวิเคราะห์จากคำตอบเสริม')
                    st.markdown(f"- คุณเลือก “{ans1}” → {text1}")
                    st.markdown(f"- คุณกลัว “{ans2}” → {text2}")
