import streamlit as st
import json
import pandas as pd

def get_results(epoch) -> dict:
    results = json.load(open("deploy_data/json/result/gru/model_" + str(epoch) + ".json"))
    return results

def get_caption(epoch, video_id) -> str:
    results = get_results(epoch)
    return results["predictions"][video_id][0]["caption"]

def get_score(epoch, video_id) -> str:
    results = get_results(epoch)
    return list(results["scores"].values())


def get_groundtruth_list(video_id) -> list:
    groundtruth_list = []
    for s in groundtruth_all["sentences"] :
        if s["video_id"] == video_id :
            groundtruth_list.append(s["caption"])
    return groundtruth_list

def get_groundtruth_sample(video_id) -> str:
    for s in groundtruth_all["sentences"] :
        if s["video_id"] == video_id :
            return s["caption"]

def get_caption_and_score_list(video_id):
    caption_list = []
    score_list = []
    for c in checkpoint_list :
        j = json.load(open("deploy_data/json/result/gru/model_" + str(c) + ".json"))
        caption = j["predictions"][video_id][0]["caption"]
        caption_list.append(caption)
        score = list(j["scores"].values())
        # score_list.append(score)
        score_list.append(score[3:7])
    return caption_list, score_list

def main() -> None:
    st.sidebar.title("Video Captioning Demo")
    st.sidebar.markdown("B4 Takuto Murase  2022/07/01")

    with st.sidebar:
        st.markdown("# How to use")
        st.markdown("1. Select a video from the list.")
        st.markdown("2. Slide and choose epochs.")
        st.markdown("3. Generated caption and score will be shown.")

        # st.markdown("# Links")
        # st.markdown("[streamlit](https://streamlit.io/)")
        # st.markdown("[Github](https://streamlit.io/)")
        # st.markdown("[Dataset](https://streamlit.io/)")

    st.header("Settings")
   
    # space1, col2 = st.columns([1, 11])
    space1, col1, col2 = st.columns([1, 5, 6])
    with st.container():
        with space1:
            st.empty()

        with col1:
        #     st.subheader("Dataset")
        #     decoder_model = st.radio("Select a dataset", ("MSR-VTT", "VATEX(Stand By)"), disabled=True)
        #     st.subheader("Encoder Model")
        #     encoder_model = st.radio("Select a encoder model", ("CNN", "3D-CNN(Stand By)"), disabled=True)
        #     st.subheader("Decoder Model")
        #     decoder_model = st.radio("Select a decoder model", ("GRU", "LSTM(Stand By)"), disabled=True)
            st.subheader("Video")
            video_id = st.selectbox(
                'Select a test video',
                test_video_list
            )
            if video_id :
                video_file = open('deploy_data/video/' + video_id + '.mp4', 'rb')
                video_bytes = video_file.read()
                caption_list, score_list = get_caption_and_score_list(video_id)
                groundtruth_sample = get_groundtruth_sample(video_id)

        with col2:
            # st.subheader("Video")
            # video_id = st.selectbox(
            #     'Select a test video',
            #     test_video_list
            # )
            if video_id :
                # video_file = open('data/test_videos/TestVideo/' + video_id + '.mp4', 'rb')
                # video_bytes = video_file.read()
                # caption_list, score_list = get_caption_and_score_list(video_id)
                # groundtruth_sample = get_groundtruth_sample(video_id)
                st.video(video_bytes)
        st.subheader("Epoch")
        epoch = st.select_slider('Select an epoch', options=checkpoint_list)
        caption = get_caption(epoch, video_id)
        score = get_score(epoch, video_id)


    st.header("Results")
    space1, col1, col2 = st.columns([1, 5, 6])
    with st.container():
        with space1:
            st.empty()

        with col1:
            st.subheader("Caption")
            st.markdown("Ground Truth")
            st.markdown(groundtruth_sample)
            st.markdown("Generated")
            st.markdown("<strong><span style=color:red;>" + caption + "</span></strong>", unsafe_allow_html=True)

        with col2:
            st.subheader("Score")
            df1 = pd.DataFrame({
                "Evaluation": eval_list,
                "Score": score[3:7]
            })
            styler = df1.style.hide(axis="index")
            st.write(styler.to_html(), unsafe_allow_html=True)

    with st.container():
        space1, col1, col2 = st.columns([1, 5, 6])
        with space1:
            st.empty()

        with col1:
            # Show Captions at each epoch
            st.subheader("Generated Caption List")
            df2 = pd.DataFrame({
                "Epoch": checkpoint_list,
                "Generated Caption": caption_list,
            })
            styler = df2.style.hide(axis="index")
            st.write(styler.to_html(), unsafe_allow_html=True)

        # with space2:
        #     st.empty()

        with col2:
            # Show Score List
            st.subheader("Score Graph")

            df4 = pd.DataFrame(
                data = score_list,
                index = checkpoint_list,
                columns = eval_list,
            )
            st.line_chart(df4)

if __name__ == "__main__":
    # epoch_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    #              11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    #              21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    #              31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    #              41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    #              60, 70, 80, 90, 100, 200, 300, 400, 500,
    #              1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    st.set_page_config(
        page_title="Video Captioning Demo",
        initial_sidebar_state="expanded",
        layout="wide"
    )
    test_video_list = [
        'video7016',
        'video7202',
        'video7216',
        'video7228',
        'video7240',
        'video7243',
        'video7245',
        'video7246',
        'video7251',
        'video7254',
        'video7255',
        'video7258',
        'video7280',
    ]
    # eval_list = ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L", "CIDEr"]
    eval_list = ["BLEU_4", "METEOR", "ROUGE_L", "CIDEr"]
    checkpoint_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    groundtruth_all = json.load(open("data/test_videodatainfo.json"))

    main()
