import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------
# 1) โหลด+สลับคำถามแค่ครั้งเดียว
# -----------------------------------
@st.cache_data
def load_questions():
    df = pd.read_csv("questions.csv")
    # สุ่มคำถามครั้งเดียว (random_state คงที่)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

df_questions = load_questions()

# -----------------------------------
# 2) สำนักคำถามย่อยกลาง (สำหรับ Core ที่ใกล้กัน)
# -----------------------------------
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
            'การผิดพลาดทางศีลธรรม (1)',
            'การไม่มีใครต้องการ (2)',
            'ความล้มเหลวและไม่น่าประทับใจ (3)',
            'การไม่มีใครเข้าใจ (4)',
            'การไม่มีข้อมูลหรือถูกควบคุม (5)',
            'การไม่มีหลักยึด / โดดเดี่ยว (6)',
            'ความเบื่อ / ความเจ็บปวด (7)',
            'การไร้พลัง / ถูกครอบงำ (8)',
            'ความขัดแย้ง / ความวุ่นวาย (9)'
        ]
    }
]

# -----------------------------------
# 3) เตรียม session_state
# -----------------------------------
if "main_submitted" not in st.session_state:
    st.session_state.main_submitted = False
if "responses" not in st.session_state:
    st.session_state.responses = []
if "sub_submitted" not in st.session_state:
    st.session_state.sub_submitted = False
if "sub_answers" not in st.session_state:
    st.session_state.sub_answers = {}

# -----------------------------------
# 4) UI และ Main Form
# -----------------------------------
st.set_page_config(page_title="Enneagram Deep Test", layout="wide")
st.title("🧠 Enneagram 81-Question Test (Core / Shadow / Behavior)")

st.markdown("### โปรดให้คะแนนแต่ละข้อ (1 = ไม่ตรงเลย, 5 = ตรงมาก)")

with st.form("enneagram_form"):
    temp_responses = []
    for idx, row in df_questions.iterrows():
        score = st.slider(
            f"ข้อ {row['Question Number']}: {row['Question (Thai)'].split('สำหรับ')[0].strip()}",
            1, 5, 3,
            key=f"main_slider_{idx}"
        )
        temp_responses.append({
            "Type": row["Enneagram Type"],
            "Category": row["Question Category"],
            "Score": score
        })

    if st.form_submit_button("ประมวลผลผลลัพธ์"):
        st.session_state.main_submitted = True
        st.session_state.responses = temp_responses

# -----------------------------------
# 5) Process และโชว์ผลหลัก
# -----------------------------------
if st.session_state.main_submitted:
    df_resp = pd.DataFrame(st.session_state.responses)
    summary = df_resp.groupby(["Type", "Category"]).mean(numeric_only=True).reset_index()
    pivot_table = summary.pivot(index="Type", columns="Category", values="Score").fillna(0)

    st.markdown("## 📊 ผลสรุป")
    st.dataframe(pivot_table.style.background_gradient(cmap="YlGnBu"))

    # — Radar Chart Core —
    core_scores = pivot_table.get("Core", pd.Series(dtype=float))
    if not core_scores.empty:
        labels = [t.replace("Type ", "T") for t in core_scores.index]
        values = core_scores.values.tolist()
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]; angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        st.pyplot(fig)

        # หาค่าสูงสุด–รอง
        sorted_scores = core_scores.sort_values(ascending=False)
        top_type = sorted_scores.index[0]
        second_type = sorted_scores.index[1]
        top_score = sorted_scores.iloc[0]
        second_score = sorted_scores.iloc[1]

        type_num = top_type.split(":")[0].replace("Type ", "")
        st.success(f"คุณมีแนวโน้มเป็น **Type {type_num}** (Core) → {top_type.split(': ')[1]}")

        # — Warning ถ้าคะแนนใกล้กัน —
        if abs(top_score - second_score) < 0.2:
            st.warning(
                f"คะแนน Core ใกล้กันระหว่าง\n"
                f"- {top_type.split(': ')[1]} ({top_score:.2f})\n"
                f"- {second_type.split(': ')[1]} ({second_score:.2f})\n\n"
                "กรุณาตอบคำถามเสริมด้านล่างเพื่อช่วยแยก Core ให้ชัดขึ้น"
            )

            # -----------------------
            # 6) Sub-form ใน Expander
            # -----------------------
            with st.expander("🧠 คำถามเสริมเพื่อแยก Core"):
                with st.form("sub_form"):
                    for q in universal_sub_questions:
                        # เก็บคำตอบลง session_state.sub_answers
                        st.session_state.sub_answers[q['question']] = st.radio(
                            q['question'],
                            q['choices'],
                            key=f"sub_radio_{q['question']}"
                        )

                    if st.form_submit_button("🔍 วิเคราะห์จากคำตอบข้างต้น"):
                        st.session_state.sub_submitted = True

                # ----------
                # 7) แสดงผลวิเคราะห์เพิ่มเติม
                # ----------
                if st.session_state.sub_submitted:
                    # (ตัวอย่าง: สรุปจากคำตอบแรก)
                    ans1 = st.session_state.sub_answers[universal_sub_questions[0]['question']]
                    ans2 = st.session_state.sub_answers[universal_sub_questions[1]['question']]

                    st.markdown("### 💡 ผลวิเคราะห์จากคำตอบเสริม")
                    st.markdown(f"- คุณเลือก “{ans1}” → บ่งชี้ถึงการตัดสินใจโดย…")
                    st.markdown(f"- คุณกลัว “{ans2}” → บ่งชี้ว่าคุณให้ความสำคัญกับ…")
