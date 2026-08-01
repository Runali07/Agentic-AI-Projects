import streamlit as st
from langgraph.types import Command
from Human_in_the_loop import app
st.set_page_config(
    page_title="LinkedIn Post Generator",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI LinkedIn Post Generator")
st.write("Generate, review and improve LinkedIn posts using Human-in-the-Loop.")



if "started" not in st.session_state:
    st.session_state.started = False

if "result" not in st.session_state:
    st.session_state.result = None

if "config" not in st.session_state:
    st.session_state.config = {
        "configurable": {
            "thread_id": "linkedin_session"
        }
    }


topic = st.text_input(
    "Enter your topic",
    placeholder="Example: Role of confidence in corporate success"
)


if st.button("Generate Post"):

    if topic.strip() == "":
        st.warning("Please enter a topic.")
        st.stop()

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    st.session_state.result = app.invoke(
        initial_state,
        config=st.session_state.config
    )

    st.session_state.started = True


if st.session_state.started:

    result = st.session_state.result

    if "__interrupt__" in result:

        interrupt_data = result["__interrupt__"][0].value

        st.subheader(
            f"Draft (Attempt {interrupt_data['attempt']})"
        )

        st.text_area(
            "Generated LinkedIn Post",
            interrupt_data["draft"],
            height=350
        )

        feedback = st.text_area(
            "Reviewer Feedback",
            placeholder="Type approved OR give feedback..."
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button("Approve"):

                st.session_state.result = app.invoke(
                    Command(resume="approved"),
                    config=st.session_state.config
                )

                st.rerun()

        with col2:

            if st.button("Submit Feedback"):

                if feedback.strip() == "":
                    st.warning("Please enter feedback.")
                else:

                    st.session_state.result = app.invoke(
                        Command(resume=feedback),
                        config=st.session_state.config
                    )

                    st.rerun()

    else:

        st.success("LinkedIn Post Generated Successfully!")

        st.subheader("Final LinkedIn Post")

        st.text_area(
            "",
            result["draft"],
            height=350
        )

        st.metric(
            "Total Attempts",
            result["attempt"]
        )

        st.metric(
            "Approved",
            str(result["is_approved"])
        )