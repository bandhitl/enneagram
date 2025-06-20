import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- CONFIG ---------- #
st.set_page_config(page_title="Enneagram Deep Test", layout="wide")
st.title("🧠 Enneagram 81-Question Test (Core / Shadow / Behavior)")

# ---------- INITIAL DATA ---------- #
def load_questions():
    df = pd.read_csv("questions.csv")  # Required CSV file
    return df

df_questions = load_questions()

# ---------- Universal Sub-questions ---------- #
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

# ---------- USER INPUT ---------- #
st.markdown("### โปรดให้คะแนนแต่ละข้อ (1 = ไม่ตรงเลย, 5 = ตรงมาก)")
responses = []

with st.form("enneagram_form"):
    for idx, row in df_questions.iterrows():
        score = st.slider(f"{row['Question Number']}. {row['Question (Thai)']}", 1, 5, 3)
        responses.append({
            "Type": row["Enneagram Type"],
            "Category": row["Question Category"],
            "Score": score
        })
    submitted = st.form_submit_button("ประมวลผลผลลัพธ์")

# ---------- PROCESSING ---------- #
if submitted:
    df_resp = pd.DataFrame(responses)
    summary = df_resp.groupby(["Type", "Category"]).mean(numeric_only=True).reset_index()
    pivot_table = summary.pivot(index="Type", columns="Category", values="Score").fillna(0)

    st.markdown("## 📊 ผลสรุป")
    st.dataframe(pivot_table.style.background_gradient(cmap="YlGnBu"))

    # Radar Chart for Core values
    st.markdown("### 📌 Core Radar Chart")
    core_scores = pivot_table["Core"] if "Core" in pivot_table else pd.Series()

    if not core_scores.empty:
        labels = [t.replace("Type ", "T") for t in core_scores.index]
        values = core_scores.values.tolist()
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        st.pyplot(fig)

        # Analysis
        sorted_scores = core_scores.sort_values(ascending=False)
        top_type = sorted_scores.index[0]
        second_type = sorted_scores.index[1]
        third_type = sorted_scores.index[2]
        top_score = sorted_scores.iloc[0]
        second_score = sorted_scores.iloc[1]
        third_score = sorted_scores.iloc[2]

        type_number = top_type.split(":")[0].replace("Type ", "")
        st.success(f"คุณมีแนวโน้มเป็น **Type {type_number}** มากที่สุด (Core) → {top_type.split(': ')[1]}")

        # Wing detection
        type_index = int(type_number)
        wing_candidates = [f"Type {(type_index - 1) or 9}:" , f"Type {(type_index % 9) + 1}:"]
        wing_scores = core_scores.loc[core_scores.index.str.startswith(tuple(wing_candidates))]
        if not wing_scores.empty:
            wing_type = wing_scores.idxmax()
            wing_num = wing_type.split(":")[0].replace("Type ", "")
            st.info(f"Wing ที่เป็นไปได้มากที่สุด: **Type {wing_num}** → {wing_type.split(': ')[1]}")

        # Core closeness warning: Top 2
        if abs(top_score - second_score) < 0.2:
            st.warning(f"🔍 คะแนน Core ของคุณใกล้เคียงกันระหว่าง\n- {top_type.split(': ')[1]} ({top_score:.2f})\n- {second_type.split(': ')[1]} ({second_score:.2f})\n\nกรุณาสังเกตความคิด/พฤติกรรมตนเองเพิ่มเติม เพื่อระบุ Core ที่แท้จริง")
            with st.expander("🧠 คำถามกลางเพื่อช่วยคุณแยก Core ตัวตนที่แท้จริง"):
                for q in universal_sub_questions:
                    st.radio(q['question'], q['choices'], key=q['question'])

        # Core closeness warning: Top 3
        if abs(top_score - third_score) < 0.25:
            st.warning(f"🔎 คุณมีแนวโน้ม Core ผสมจาก 3 ประเภทหลัก:\n- {top_type.split(': ')[1]} ({top_score:.2f})\n- {second_type.split(': ')[1]} ({second_score:.2f})\n- {third_type.split(': ')[1]} ({third_score:.2f})\n\nแนะนำให้สังเกตแรงผลักดันลึกสุดของตนเอง หรือพิจารณาบทบาทในสถานการณ์ต่าง ๆ เพื่อแยกความชัดเจน")

    st.markdown("### 🔎 คำอธิบาย")
    st.markdown("- **Core** = แรงผลักดันภายในแท้จริง\n- **Shadow** = จุดเปราะ จุดกลัว\n- **Behavior** = การแสดงออกที่คนอื่นเห็น")
    st.bar_chart(pivot_table)
    st.success("กราฟนี้ช่วยให้คุณเห็นสมดุลระหว่าง Core, Shadow, Behavior และค้นพบแนวโน้ม Wing")
