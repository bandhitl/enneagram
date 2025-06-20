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
    # Q1
    'ความถูกต้อง / จริยธรรม (Type 1)':       'คุณตัดสินใจโดยคำนึงถึงหลักจริยธรรมและความถูกต้องเป็นหลัก',
    'การได้รับการรักหรือยอมรับ (Type 2)':    'คุณมักตัดสินใจโดยคำนึงถึงความสัมพันธ์และความรู้สึกของผู้อื่น',
    'เป้าหมาย / ความสำเร็จ (Type 3)':       'คุณเลือกทางที่ส่งเสริมเป้าหมายและภาพลักษณ์แห่งความสำเร็จ',
    'อารมณ์ภายในและเอกลักษณ์ (Type 4)':    'คุณให้ความสำคัญกับความรู้สึกภายในและความเป็นตัวของตัวเอง',
    'ความรู้ ความเข้าใจลึก (Type 5)':        'คุณเน้นหาข้อมูลและความเข้าใจเชิงลึกก่อนตัดสินใจ',
    'ความปลอดภัยและการวางแผน (Type 6)':   'คุณวางแผนและเตรียมพร้อมเพื่อสร้างความมั่นคง',
    'อิสระและความสนุก (Type 7)':           'คุณมองหาความสนุกและอิสระเป็นหลักในการตัดสินใจ',
    'พลังอำนาจและการควบคุม (Type 8)':      'คุณเลือกเส้นทางที่ให้คุณควบคุมสถานการณ์และเข้มแข็ง',
    'ความสงบและหลีกเลี่ยงความขัดแย้ง (Type 9)': 'คุณเลือกความสงบและหลีกเลี่ยงความขัดแย้งเป็นสำคัญ',
    # Q2
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

# ---------- TYPE PROFILE MAPPING ---------- #
type_profile_map = {
    '1': 'ตรงไปตรงมา ซื่อสัตย์ เคร่งครัดเรื่องคุณภาพ ไม่ยอมปล่อยผ่านงานที่บกพร่อง',
    '2': 'เอาใจเก่ง ใส่ใจคนรอบข้าง จนบางครั้งลืมดูแลตัวเอง',
    '3': 'มุ่งมั่นผลลัพธ์ รวดเร็ว แก้ปัญหาตามเป้า แต่บางทีกดดันทีมสูง',
    '4': 'สร้างสรรค์ มีอารมณ์ลึกซึ้ง แต่งานอาจสะดุดหากอารมณ์แปรปรวน',
    '5': 'ชอบวิเคราะห์ ถามเยอะ มีมุมมองเฉียบคม แต่บางทีดูเนิร์ดเกินไป',
    '6': 'รอบคอบ เตรียมพร้อมดีมาก แต่มักกังวลจนตัดสินใจช้า',
    '7': 'กระฉับกระเฉง ริเริ่มไอเดียใหม่ๆ แต่บางครั้งสมาธิสั้น',
    '8': 'เด็ดขาด กล้าแสดงออก แต่บางทีอาจดูเกรี้ยวกราด',
    '9': 'ใจเย็น สร้างบรรยากาศสบายๆ แต่บางทีก็เฉื่อยและหลีกเลี่ยงปัญหา'
}

# ---------- SESSION STATE ---------- #
for key in ['main_submitted','sub_submitted','responses','sub_answers']:
    if key not in st.session_state:
        st.session_state[key] = False if 'submitted' in key else []

# ---------- MAIN FORM ---------- #
st.markdown("### โปรดให้คะแนนแต่ละข้อ (1 = ไม่ตรงเลย, 5 = ตรงมาก)")
with st.form('main_form'):
    temp_responses = []
    for idx, row in df_questions.iterrows():
        score = st.slider(
            f"ข้อ {row['Question Number']}: {row['Question (Thai)'].split('สำหรับ')[0].strip()}",
            1,5,3, key=f"main_{idx}"
        )
        temp_responses.append({'Type':row['Enneagram Type'],'Category':row['Question Category'],'Score':score})
    if st.form_submit_button('ประมวลผลผลลัพธ์'):
        st.session_state.main_submitted = True
        st.session_state.responses = temp_responses

# ---------- SHOW RESULTS ---------- #
if st.session_state.main_submitted:
    df_resp = pd.DataFrame(st.session_state.responses)
    summary = df_resp.groupby(['Type','Category']).mean(numeric_only=True).reset_index()
    pivot = summary.pivot(index='Type',columns='Category',values='Score').fillna(0)

    # Top & second
    core = pivot.get('Core',pd.Series(dtype=float))
    sorted_core = core.sort_values(ascending=False)
    top, second = sorted_core.index[0], sorted_core.index[1]
    top_score, second_score = sorted_core.iloc[0], sorted_core.iloc[1]

    # Display main
    num = top.split(':')[0].replace('Type ','')
    label = top.split(': ')[1]
    st.success(f"คุณมีแนวโน้มเป็น Type {num} → {label}")
    # Profile description
    desc = type_profile_map.get(num, '')
    st.markdown(f"**ลักษณะเด่น:** {desc}")

    # Collaborators & caution
    collab = sorted_core.iloc[1:3].index
    caution = sorted_core.iloc[-2:].index
    st.markdown('#### 🤝 คนที่คุณทำงานด้วยแล้วเวิร์ค')
    for t in collab:
        n,l = t.split(': ')
        st.markdown(f"- **{n}** ({l})")
    st.markdown('#### ⚠️ คนที่ควรระมัดระวังในการทำงานด้วย')
    for t in caution:
        n,l = t.split(': ')
        st.markdown(f"- **{n}** ({l})")

    # Radar chart
    labels = [t.replace('Type ','T') for t in core.index]
    vals = core.tolist(); angles = np.linspace(0,2*np.pi,len(vals),endpoint=False).tolist()
    vals+=vals[:1]; angles+=angles[:1]
    fig,ax = plt.subplots(figsize=(5,5),subplot_kw={'polar':True})
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.3)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    st.pyplot(fig)

    # Sub questions if tied
    if abs(top_score-second_score)<0.2:
        st.warning('คะแนน Core ใกล้กัน! ตอบคำถามเสริมเพื่อแยกให้ชัด')
        with st.expander('🧠 คำถามเสริม'):
            with st.form('sub'):
                for q in universal_sub_questions:
                    st.session_state.sub_answers[q['question']] = st.radio(q['question'],q['choices'],key=q['question'])
                if st.form_submit_button('วิเคราะห์'): st.session_state.sub_submitted=True
            if st.session_state.sub_submitted:
                a1=st.session_state.sub_answers[universal_sub_questions[0]['question']]
                a2=st.session_state.sub_answers[universal_sub_questions[1]['question']]
                t1=sub_analysis_map.get(a1,''); t2=sub_analysis_map.get(a2,'')
                st.markdown(f"- คุณเลือก '{a1}' → {t1}")
                st.markdown(f"- คุณกลัว '{a2}' → {t2}")
