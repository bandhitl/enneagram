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
    {'question': 'คุณคิดว่าตัวเองตัดสินใจโดยอิงจากอะไรเป็นหลัก?', 'choices': [
        'ความถูกต้อง / จริยธรรม (Type 1)', 'การได้รับการรักหรือยอมรับ (Type 2)',
        'เป้าหมาย / ความสำเร็จ (Type 3)', 'อารมณ์ภายในและเอกลักษณ์ (Type 4)',
        'ความรู้ ความเข้าใจลึก (Type 5)', 'ความปลอดภัยและการวางแผน (Type 6)',
        'อิสระและความสนุก (Type 7)', 'พลังอำนาจและการควบคุม (Type 8)',
        'ความสงบและหลีกเลี่ยงความขัดแย้ง (Type 9)'
    ]},
    {'question': 'คุณกลัวสิ่งใดมากที่สุดในชีวิต?', 'choices': [
        'การผิดพลาดทางศีลธรรม (Type 1)', 'การไม่มีใครต้องการ (Type 2)',
        'ความล้มเหลวและไม่น่าประทับใจ (Type 3)', 'การไม่มีใครเข้าใจ (Type 4)',
        'การไม่มีข้อมูลหรือถูกควบคุม (Type 5)', 'การไม่มีหลักยึด / โดดเดี่ยว (Type 6)',
        'ความเบื่อ / ความเจ็บปวด (Type 7)', 'การไร้พลัง / ถูกครอบงำ (Type 8)',
        'ความขัดแย้ง / ความวุ่นวาย (Type 9)'
    ]}
]

# ---------- MAPPING FOR SUB-ANALYSIS ---------- #
sub_analysis_map = {
    # Q1
    'ความถูกต้อง / จริยธรรม (Type 1)': 'คุณตัดสินใจโดยคำนึงถึงหลักจริยธรรมและความถูกต้องเป็นหลัก',
    'การได้รับการรักหรือยอมรับ (Type 2)': 'คุณมักตัดสินใจโดยคำนึงถึงความสัมพันธ์และความรู้สึกของผู้อื่น',
    'เป้าหมาย / ความสำเร็จ (Type 3)': 'คุณเลือกทางที่ส่งเสริมเป้าหมายและภาพลักษณ์แห่งความสำเร็จ',
    'อารมณ์ภายในและเอกลักษณ์ (Type 4)': 'คุณให้ความสำคัญกับความรู้สึกภายในและความเป็นตัวของตัวเอง',
    'ความรู้ ความเข้าใจลึก (Type 5)': 'คุณเน้นหาข้อมูลและความเข้าใจเชิงลึกก่อนตัดสินใจ',
    'ความปลอดภัยและการวางแผน (Type 6)': 'คุณวางแผนและเตรียมพร้อมเพื่อสร้างความมั่นคง',
    'อิสระและความสนุก (Type 7)': 'คุณมองหาความสนุกและอิสระเป็นหลักในการตัดสินใจ',
    'พลังอำนาจและการควบคุม (Type 8)': 'คุณเลือกเส้นทางที่ให้คุณควบคุมสถานการณ์และเข้มแข็ง',
    'ความสงบและหลีกเลี่ยงความขัดแย้ง (Type 9)': 'คุณเลือกความสงบและหลีกเลี่ยงความขัดแย้งเป็นสำคัญ',
    # Q2
    'การผิดพลาดทางศีลธรรม (Type 1)': 'คุณกลัวทำผิดจริยธรรม จึงระมัดระวังไม่ให้ล่วงเกินมาตรฐาน',
    'การไม่มีใครต้องการ (Type 2)': 'คุณกลัวไร้คุณค่า จึงดูแลความสัมพันธ์ไว้เสมอ',
    'ความล้มเหลวและไม่น่าประทับใจ (Type 3)': 'คุณกลัวไม่สำเร็จและเสียภาพลักษณ์ จึงมุ่งมั่นสูง',
    'การไม่มีใครเข้าใจ (Type 4)': 'คุณกลัวว่าความเป็นตัวเองจะไม่ถูกเข้าใจ',
    'การไม่มีข้อมูลหรือถูกควบคุม (Type 5)': 'คุณกลัวขาดข้อมูลหรือถูกจำกัด จึงเน้นเสาะหาองค์ความรู้',
    'การไม่มีหลักยึด / โดดเดี่ยว (Type 6)': 'คุณกลัวไร้ที่พึ่ง จึงเผื่อแผนและมองหาความมั่นคง',
    'ความเบื่อ / ความเจ็บปวด (Type 7)': 'คุณกลัวความเจ็บปวดและความน่าเบื่อ จึงแสวงหาความท้าทาย',
    'การไร้พลัง / ถูกครอบงำ (Type 8)': 'คุณกลัวการถูกครอบงำ จึงยืนหยัดเพื่อพลังและอิสรภาพ',
    'ความขัดแย้ง / ความวุ่นวาย (Type 9)': 'คุณกลัวความวุ่นวาย จึงพยายามสร้างความสมดุลและสันติ'
}

# ---------- PROFILE BULLETS ---------- #
type_profile_map = {
    '1': [
        'ตรงไปตรงมา ไม่ยอมปล่อยผ่านงานบกพร่อง',
        'ตรวจงานละเอียดจนคนรอบข้างกดดัน',
        'ตั้งมาตรฐานสูงมาก ทุกอย่างต้องสมบูรณ์',
        'หงุดหงิดเมื่อเห็นความผิดแม้เล็กน้อย',
        'เสียงดังเตือนเวลาเจอคนทำงานไม่ละเอียด'
    ],
    '2': [
        'เอาใจเก่ง ดูแลคนรอบข้างจนเหมือนไม่มีวันหยุด',
        'ชอบช่วยเหลือเกินขอบเขต บางครั้งตัวเองเหนื่อย',
        'อยากให้ทุกคนรู้สึกดี ไม่ชอบบรรยากาศตึงเครียด',
        'ทำงานเป็นทีมดี แต่ถ้ารู้สึกถูกเมินจะเครียด',
        'บางทีลืมถามตัวเองว่าต้องการอะไร'
    ],
    # ... profiles for 3–9 omitted for brevity
}

# ---------- SESSION STATE ---------- #
for key in ['main_submitted', 'sub_submitted']:
    if key not in st.session_state:
        st.session_state[key] = False
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'sub_answers' not in st.session_state:
    st.session_state.sub_answers = {}

# ---------- MAIN FORM ---------- #
st.markdown("### โปรดให้คะแนนแต่ละข้อ (1 = ไม่ตรงเลย, 5 = ตรงมาก)")
with st.form('main_form'):
    temp = []
    for i, r in df_questions.iterrows():
        val = st.slider(f"ข้อ {r['Question Number']}: {r['Question (Thai)'].split('สำหรับ')[0].strip()}",1,5,3,key=f"s{i}")
        temp.append({'Type':r['Enneagram Type'],'Category':r['Question Category'],'Score':val})
    if st.form_submit_button('ประมวลผลผลลัพธ์'):
        st.session_state.main_submitted=True
        st.session_state.responses=temp

# ---------- RESULTS ---------- #
if st.session_state.main_submitted:
    df_r = pd.DataFrame(st.session_state.responses)
    sum_r = df_r.groupby(['Type','Category']).mean(numeric_only=True).reset_index()
    piv = sum_r.pivot(index='Type',columns='Category',values='Score').fillna(0)
    core = piv.get('Core',pd.Series(dtype=float))
    so = core.sort_values(ascending=False)
    top, sec = so.index[0], so.index[1]
    ts, ss = so.iloc[0], so.iloc[1]
    num, lbl = top.replace('Type ','').split(':')[0], top.split(': ')[1]

    st.success(f"คุณมีแนวโน้มเป็น Type {num} → {lbl}")
    # bullets
    st.markdown('**ลักษณะเด่น (5 ข้อที่ใช่เลย):**')
    for b in type_profile_map.get(num,[]): st.markdown(f"- {b}")

    # collaborators & caution
    coll = so.iloc[1:3].index
    caut = so.iloc[-2:].index
    st.markdown('#### 🤝 คนที่คุณทำงานด้วยแล้วเวิร์ค')
    for t in coll: n,l=t.split(': '); st.markdown(f"- **{n}** ({l})")
    st.markdown('#### ⚠️ คนที่ควรระมัดระวังในการทำงานด้วย')
    for t in caut: n,l=t.split(': '); st.markdown(f"- **{n}** ({l})")

    # radar chart
    labels=[x.replace('Type ','T') for x in core.index]
    vals=list(core); ang=np.linspace(0,2*np.pi,len(vals),endpoint=False).tolist()
    vals+=vals[:1]; ang+=ang[:1]
    fig,ax=plt.subplots(subplot_kw={'polar':True})
    ax.plot(ang,vals,linewidth=2,linestyle='solid'); ax.fill(ang,vals,alpha=0.25)
    ax.set_thetagrids(np.degrees(ang[:-1]),labels)
    st.pyplot(fig)

    # sub-questions if tie
    if abs(ts-ss)<0.2:
        st.warning('คะแนน Core ใกล้กัน! ตอบคำถามเสริมเพื่อแยกให้ชัด')
        with st.expander('🧠 คำถามเสริมเพื่อแยก Core'):
            with st.form('sub'): # subform
                for q in universal_sub_questions:
                    st.session_state.sub_answers[q['question']] = st.radio(q['question'],q['choices'],key=q['question'])
                if st.form_submit_button('🔍 วิเคราะห์'): st.session_state.sub_submitted=True
            if st.session_state.sub_submitted:
                a1=st.session_state.sub_answers[universal_sub_questions[0]['question']]
                a2=st.session_state.sub_answers[universal_sub_questions[1]['question']]
                st.markdown(f"- คุณเลือก '{a1}' → {sub_analysis_map.get(a1,'')}")
                st.markdown(f"- คุณกลัว '{a2}' → {sub_analysis_map.get(a2,'')}")
