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

df_questions = load_questions().sample(frac=1, random_state=42).reset_index(drop=True)

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
        score = st.slider(f"ข้อ {row['Question Number']}: {row['Question (Thai)'].split('สำหรับ')[0].strip()}", 1, 5, 3)
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
        wing_candidates = [f"Type {(type_index - 1) or 9}:", f"Type {(type_index % 9) + 1}:"]
        wing_scores = core_scores.loc[core_scores.index.str.startswith(tuple(wing_candidates))]
        wing_description = ""
        if not wing_scores.empty:
            wing_type = wing_scores.idxmax()
            wing_num = wing_type.split(":")[0].replace("Type ", "")
            wing_label = wing_type.split(": ")[1]
            st.info(f"Wing ที่เป็นไปได้มากที่สุด: **Type {wing_num}** → {wing_label}")

            # สรุป Core + Wing
            core_label = top_type.split(": ")[1]
            st.markdown("---")
            st.markdown(f"### 🧬 บุคลิกภาพโดยรวมของคุณ: **{core_label}** with Wing **{wing_label}**")

            summary_map = {
                "The Reformer": {
                    "strength": "ยึดมั่นในความถูกต้อง ซื่อสัตย์ จริงจัง",
                    "stress": "ไม่ชอบความผิดพลาด และกดดันตัวเอง",
                    "conflict": "มักไม่สบายใจกับคนที่ชอบเลื่อนมาตรฐาน หรือไม่ใส่ใจรายละเอียด",
                    "synergy": "ทำงานเข้ากับคนตรงไปตรงมา มีระบบ หรือคนที่เคารพกฎเหมือนกัน"
                },
                "The Helper": {
                    "strength": "เห็นอกเห็นใจ เป็นผู้สนับสนุน",
                    "stress": "รู้สึกไร้ค่าเมื่อไม่มีคนต้องการ",
                    "conflict": "ไม่สบายใจกับคนที่เย็นชา ไม่เห็นคุณค่า",
                    "synergy": "เข้ากับคนที่เปิดใจ รับรู้คุณค่าที่คุณให้ได้"
                },
                "The Achiever": {
                    "strength": "มีเป้าหมาย ชอบความสำเร็จ",
                    "stress": "กลัวล้มเหลว และยึดภาพลักษณ์",
                    "conflict": "ไม่สบายใจกับคนช้า หรือขาดประสิทธิภาพ",
                    "synergy": "เข้ากับคนขยัน คล่องตัว และมองเป้าหมายร่วม"
                },
                "The Individualist": {
                    "strength": "ลึกซึ้ง มีเอกลักษณ์",
                    "stress": "อ่อนไหวง่าย และรู้สึกไม่เหมือนใคร",
                    "conflict": "ไม่สบายใจกับคนที่ตื้นเขิน หรือเหมารวม",
                    "synergy": "เข้ากับคนที่เปิดใจ ยอมรับความต่าง และลึกซึ้งเหมือนกัน"
                },
                "The Investigator": {
                    "strength": "ใฝ่รู้ ลึกซึ้ง ช่างสังเกต",
                    "stress": "ถอยตัวจากโลกเมื่อตึงเครียด",
                    "conflict": "ไม่สบายใจกับคนที่เร่งเร้า หรือขัดจังหวะความคิด",
                    "synergy": "เข้ากับคนที่ให้พื้นที่ และเคารพการคิดวิเคราะห์"
                },
                "The Loyalist": {
                    "strength": "ระมัดระวัง ซื่อสัตย์ วางแผนเก่ง",
                    "stress": "วิตกกังวลง่าย และลังเล",
                    "conflict": "ไม่สบายใจกับคนที่ไม่น่าไว้ใจ หรือเปลี่ยนใจบ่อย",
                    "synergy": "เข้ากับคนที่เสถียร เชื่อถือได้ และวางแผนร่วม"
                },
                "The Enthusiast": {
                    "strength": "คิดเร็ว ร่าเริง มองโลกบวก",
                    "stress": "เบื่อง่าย และหลีกเลี่ยงความเจ็บปวด",
                    "conflict": "ไม่สบายใจกับคนที่ยึดติด กดดัน หรือเน้นความเคร่งครัด",
                    "synergy": "เข้ากับคนที่ยืดหยุ่น เปิดรับ และร่วมสนุก"
                },
                "The Challenger": {
                    "strength": "เด็ดขาด เข้มแข็ง ปกป้องผู้อื่น",
                    "stress": "ไม่ยอมแพ้ และต่อต้านการควบคุม",
                    "conflict": "ไม่สบายใจกับคนที่อ่อนแอ หลอกลวง หรือเล่นเกม",
                    "synergy": "เข้ากับคนที่ตรงไปตรงมา และมั่นใจในตัวเอง"
                },
                "The Peacemaker": {
                    "strength": "รักสงบ ประนีประนอม เห็นใจทุกฝ่าย",
                    "stress": "เพิกเฉย หลีกเลี่ยงปัญหา",
                    "conflict": "ไม่สบายใจกับคนที่ขัดแย้งเก่ง กดดัน หรือเสียงดัง",
                    "synergy": "เข้ากับคนที่ให้พื้นที่ ปรับตัวได้ และไม่เร่งเร้า"
                }
            }

            if core_label in summary_map:
                summary = summary_map[core_label]
                st.markdown(f"#### จุดแข็ง: {summary['strength']}")
                st.markdown(f"#### ความเครียดเมื่อเผชิญปัญหา: {summary['stress']}")
                st.markdown(f"#### ไม่สบายใจเมื่อทำงานกับคนแบบ: {summary['conflict']}")
                st.markdown(f"#### ทำงานเข้าขากับคนแบบ: {summary['synergy']}")
        # Core closeness warning: Top 2
        if abs(top_score - second_score) < 0.2:
            st.warning(f"🔍 คะแนน Core ของคุณใกล้เคียงกันระหว่าง\n- {top_type.split(': ')[1]} ({top_score:.2f})\n- {second_type.split(': ')[1]} ({second_score:.2f})\n\nกรุณาสังเกตความคิด/พฤติกรรมตนเองเพิ่มเติม เพื่อระบุ Core ที่แท้จริง")
            with st.expander("🧠 คำถามกลางเพื่อช่วยคุณแยก Core ตัวตนที่แท้จริง"):
            answers = {}
            for q in universal_sub_questions:
                answers[q['question']] = st.radio(q['question'], q['choices'], key=q['question'])
            submit_q = st.button("🔍 วิเคราะห์จากคำตอบข้างต้น")


