import streamlit as st 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import torch 
import torchvision 
from torchvision import models, transforms
from model import predict
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("画像認識アプリ")
st.sidebar.write("画像判定を行う")

st.sidebar.title("")

img_source = st.sidebar.radio("画像のソースを選択してください",
("画像をアップロード","カメラで撮影"))
if img_source =="画像をアップロード":
  img_file = st.sidebar.file_uploader("画像を選択してください", type=["jpeg","png","jpg"])
elif img_source =="カメラで撮影":
  img_file = st.camera_input("カメラで撮影")

if img_file is not None:
  with st.spinner("推定中..."):
    img= Image.open(img_file)
    st.image(img, caption="対象の画像", width= 480)
    st.write("")

    results = predict(img)

    st.subheader("判定結果")
    n_top = 5
    for result in results[:n_top]:
      st.write(str(round(result[1]*100, 2))+ "%の確率で" + result[0] + "です")

    pie_labels = [result[0] for result in results[:n_top]]
    pie_labels.append("others")
    pie_probs = [result[1] for result in results[:n_top:]]
    pie_probs.append(sum([result[1] for result in results[n_top:]]))
    fig, ax = plt.subplots()
    wedgeprops = {"width":0.3, "edgecolor":'white'}
    textprops = {"fontsize":6}
    ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle= 90,
    textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)
    st.pyplot(fig)






