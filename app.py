import streamlit as st, numpy as np, cv2, torch, timm, pandas as pd
from PIL import Image
import torchvision.transforms as T

IMG_SIZE, THR = 256, 0.6
LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
tf = T.Compose([T.ToPILImage(), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def crop_black(img, tol=7):
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > tol
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    if len(rows) == 0 or len(cols) == 0: return img
    return img[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

def pad_square(img):
    h, w = img.shape[:2]; s = max(h, w)
    out = np.zeros((s, s, 3), dtype=np.uint8)
    out[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = img
    return out

def preprocess(img):
    img = cv2.resize(pad_square(crop_black(img)), (IMG_SIZE, IMG_SIZE))
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), IMG_SIZE/30), -4, 128)

class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)
    def _fwd(self, m, i, o):
        self.acts = o
        if o.requires_grad:
            o.register_hook(lambda g: setattr(self, 'grads', g))
    def __call__(self, x):
        out = self.model(x)
        cls = out.argmax(1).item()
        self.model.zero_grad(); out[0, cls].backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1)).squeeze().detach().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.softmax(out, 1).detach().numpy()[0], cls

@st.cache_resource
def load_model():
    m = timm.create_model('efficientnet_b0', pretrained=False, num_classes=5, drop_rate=0.3)
    m.load_state_dict(torch.load('effnet_b0.pth', map_location='cpu'))
    m.eval()
    return m, GradCAM(m, m.bn2)

def overlay(img, cam):
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 0.55, heat, 0.45, 0)

def triage(prob):
    conf, ref = float(prob.max()), float(prob[2:].sum())
    if conf < THR: return 'uncertain', 'UNCERTAIN: please consult an eye doctor'
    if ref >= 0.5: return 'referable', 'REFERABLE DR: refer to an eye specialist'
    return 'normal', 'No referable DR detected: routine annual screening'

st.set_page_config(page_title='Retina DR Screening', page_icon='👁️', layout='wide')
st.title('👁️ Diabetic Retinopathy Screening Assistant')
st.caption('SDG 3: Good Health and Well-being | IBM SkillsBuild ML & Applied AI Internship')
st.warning('Screening-support prototype only. Not a medical diagnosis. Always consult a qualified eye doctor.')

model, gradcam = load_model()
file = st.file_uploader('Upload a retina (fundus) image', type=['png', 'jpg', 'jpeg'])

if file:
    raw = np.array(Image.open(file).convert('RGB'))
    proc = preprocess(raw)
    cam, prob, cls = gradcam(tf(proc).unsqueeze(0))
    kind, msg = triage(prob)

    c1, c2, c3 = st.columns(3)
    c1.image(raw, caption='Uploaded image', use_container_width=True)
    c2.image(proc, caption='Preprocessed', use_container_width=True)
    c3.image(overlay(proc, cam), caption='Grad-CAM: areas the model focused on', use_container_width=True)

    {'normal': st.success, 'uncertain': st.warning, 'referable': st.error}[kind](msg)
    st.subheader(f'Predicted grade: {LABELS[cls]} ({prob[cls]*100:.1f}% confidence)')
    st.bar_chart(pd.DataFrame({'Probability': prob}, index=LABELS))
    st.caption('Model: EfficientNet-B0 (transfer learning) | Test QWK 0.89 | Referable-DR sensitivity 0.94, specificity 0.95')