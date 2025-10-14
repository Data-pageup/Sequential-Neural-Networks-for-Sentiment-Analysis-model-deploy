import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Sequential Models for Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(120deg, #00d4ff, #0099ff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            filter: drop-shadow(0 0 5px rgba(0, 212, 255, 0.5));
        }
        to {
            filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.8));
        }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #00d4ff;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 3rem;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #00d4ff, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        border-left: 5px solid #00d4ff;
        padding-left: 1rem;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 0;
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, transparent);
    }
    
    .metric-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 153, 255, 0.1));
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        border-color: #00d4ff;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
    }
    
    .info-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .success-card {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 212, 255, 0.1));
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #00ff88;
        margin: 1rem 0;
    }
    
    .error-card {
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.1), rgba(255, 0, 0, 0.1));
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 152, 0, 0.1));
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #0099ff);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid rgba(0, 212, 255, 0.2);
    }
    
    div[data-testid="stSidebar"] .stRadio > label {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .highlight-text {
        color: #00d4ff;
        font-weight: 600;
    }
    
    .code-block {
        background: rgba(0, 0, 0, 0.4);
        border-left: 3px solid #00d4ff;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
    }
    
    .stats-number {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h1, h2, h3 {
        color: #ffffff;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 153, 255, 0.1));
        border-radius: 10px;
        padding: 1rem 2rem;
        color: #ffffff;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff, #0099ff);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Sequential Neural Networks for Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">RNN vs LSTM vs GRU: A Comprehensive Comparison</div>', unsafe_allow_html=True)

# Sidebar navigation with styling
st.sidebar.markdown("## 🧭 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("", [
    "🏠 Project Overview",
    "🔧 Data Processing",
    "🏗️ Model Architecture",
    "📊 Training Results",
    "⚖️ Model Comparison",
    "🎯 Conclusions"
], label_visibility="collapsed")

# Training data
training_data = {
    'RNN': pd.DataFrame({
        'Epoch': [1, 2, 3, 4, 5],
        'Train Accuracy': [50.56, 49.68, 50.55, 50.34, 51.47],
        'Val Accuracy': [49.00, 49.05, 49.88, 50.33, 50.45],
        'Train Loss': [0.7248, 0.7048, 0.6967, 0.6957, 0.6946],
        'Val Loss': [0.6957, 0.6937, 0.6957, 0.6932, 0.6933]
    }),
    'LSTM': pd.DataFrame({
        'Epoch': [1, 2, 3, 4, 5],
        'Train Accuracy': [50.69, 59.50, 58.70, 81.39, 91.69],
        'Val Accuracy': [61.85, 53.07, 68.85, 87.52, 87.77],
        'Train Loss': [0.6937, 0.6666, 0.6295, 0.4224, 0.2262],
        'Val Loss': [0.6650, 0.6832, 0.5843, 0.3027, 0.3067]
    }),
    'GRU': pd.DataFrame({
        'Epoch': [1, 2, 3, 4, 5],
        'Train Accuracy': [50.03, 56.52, 89.76, 95.21, 97.81],
        'Val Accuracy': [51.78, 87.20, 89.05, 88.23, 87.63],
        'Train Loss': [0.6938, 0.6637, 0.2591, 0.1417, 0.0746],
        'Val Loss': [0.6963, 0.2983, 0.2602, 0.2932, 0.3800]
    })
}

test_results = pd.DataFrame({
    'Model': ['RNN', 'LSTM', 'GRU'],
    'Test Accuracy': [49.62, 87.42, 87.72],
    'Avg Time per Epoch': [102, 376, 292],
    'Total Training Time': [510, 1880, 1460]
})

# PAGE 1: PROJECT OVERVIEW
if page == "🏠 Project Overview":
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="stats-number">3</div>
            <div>Models Tested</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="stats-number">87.7%</div>
            <div>Best Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="stats-number">5</div>
            <div>Training Epochs</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="stats-number">50K</div>
            <div>Movie Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Problem Statement</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>🎯 Objective</h3>
        <p>Build a <span class="highlight-text">binary sentiment classifier</span> to predict whether movie reviews express positive or negative sentiment.</p>
        
        <h3>📚 Dataset</h3>
        <p><span class="highlight-text">Kaggle Movie Reviews Dataset</span></p>
        <ul>
            <li>Contains movie reviews (text) and corresponding sentiment labels</li>
            <li>Binary classification: Positive (1) or Negative (0)</li>
            <li>Real-world user-generated content with diverse vocabulary</li>
        </ul>
        
        <h3>🔬 Approach</h3>
        <p>Compare three <span class="highlight-text">sequential neural network architectures</span>:</p>
        <ul>
            <li><strong>Simple RNN</strong> - Baseline recurrent model</li>
            <li><strong>LSTM</strong> - Long Short-Term Memory networks</li>
            <li><strong>GRU</strong> - Gated Recurrent Units</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-card">
        <h3>📊 Key Metrics</h3>
        <ul>
            <li><strong>Accuracy</strong> - Correct predictions</li>
            <li><strong>Loss</strong> - Binary Crossentropy</li>
            <li><strong>Training Time</strong> - Efficiency</li>
            <li><strong>Convergence Speed</strong> - Learning rate</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Why Sequential Models?</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>🔄 Sequential Nature of Text</h3>
        <p>Text data is inherently sequential. The meaning of a sentence depends critically on:</p>
        <ul>
            <li>Word order and arrangement</li>
            <li>Context from previous words</li>
            <li>Long-range dependencies</li>
        </ul>
        <p><em>Example: "not good" vs "good" - same words, opposite meanings</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-card">
        <h3>⚠️ The Challenge</h3>
        <p><strong>Vanishing Gradient Problem</strong></p>
        <p>Simple RNNs struggle to learn long-term dependencies because:</p>
        <ul>
            <li>Gradients shrink exponentially through time</li>
            <li>Cannot remember information from many steps ago</li>
            <li>Training becomes ineffective</li>
        </ul>
        <p><strong>Solution:</strong> LSTM and GRU architectures with gating mechanisms</p>
        </div>
        """, unsafe_allow_html=True)

# PAGE 2: DATA PROCESSING
elif page == "🔧 Data Processing":
    st.markdown('<div class="section-header">Data Preprocessing Pipeline</div>', unsafe_allow_html=True)
    
    # Visual Pipeline
    st.markdown("""
    <div class="info-card">
    <h2 style="text-align: center;">📥 Raw Data → 🔧 Processing → 🎯 Model-Ready Data</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
        <h3>Step 1</h3>
        <h2>Text Cleaning</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <ul>
            <li>Convert to lowercase</li>
            <li>Remove HTML tags</li>
            <li>Remove URLs</li>
            <li>Remove numbers</li>
            <li>Remove punctuation</li>
            <li>Remove stopwords</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
        <h3>Step 2</h3>
        <h2>Tokenization</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <ul>
            <li>Build vocabulary (10,000 words)</li>
            <li>Convert words to integers</li>
            <li>Create word-to-index mapping</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
        <h3>Step 3</h3>
        <h2>Padding</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <ul>
            <li>Fixed sequence length: 200</li>
            <li>Pad short sequences with zeros</li>
            <li>Truncate long sequences</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Transformation Example</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="error-card">
        <h3>❌ Before Cleaning</h3>
        <div class="code-block">
        "The movie was GREAT!!! 🎬<br>
        I LOVED it SO much.<br>
        Visit www.example.com for more.<br>
        Rating: 10/10"
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-card">
        <h3>✅ After Cleaning</h3>
        <div class="code-block">
        "movie great loved much"
        </div>
        <br>
        <h3>✅ After Tokenization</h3>
        <div class="code-block">
        [45, 892, 234, 567]
        </div>
        <br>
        <h3>✅ After Padding (200 tokens)</h3>
        <div class="code-block">
        [45, 892, 234, 567, 0, 0, 0, ...]
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Dataset Split</div>', unsafe_allow_html=True)
    
    # Visual representation of split
    split_data = pd.DataFrame({
        'Dataset': ['Training', 'Testing'],
        'Percentage': [80, 20],
        'Reviews': [40000, 10000]
    })
    
    fig = go.Figure(data=[go.Pie(
        labels=split_data['Dataset'],
        values=split_data['Percentage'],
        hole=.4,
        marker=dict(colors=['#00d4ff', '#0099ff']),
        textfont=dict(size=18, color='white')
    )])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-card">
    <h3>🎯 Stratified Split</h3>
    <p>Maintains the same proportion of positive and negative reviews in both training and test sets to ensure representative evaluation.</p>
    </div>
    """, unsafe_allow_html=True)

# PAGE 3: MODEL ARCHITECTURE
elif page == "🏗️ Model Architecture":
    st.markdown('<div class="section-header">Neural Network Architectures</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔴 Simple RNN", "🟢 LSTM", "🔵 GRU"])
    
    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            <div class="error-card">
            <h2>Simple RNN Architecture</h2>
            <div class="code-block">
Model: Simple RNN
_________________________________________________
Layer (type)              Output Shape       Params
=================================================
embedding                 (None, 200, 128)   1,280,000
simple_rnn                (None, 64)         12,352
dropout (0.5)             (None, 64)         0
dense + sigmoid           (None, 1)          65
=================================================
Total params: 1,292,417
Trainable params: 1,292,417
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
            <h3>How Simple RNN Works</h3>
            <ul>
                <li>Maintains a single hidden state vector</li>
                <li>Updates state at each time step: h(t) = tanh(W × [h(t-1), x(t)])</li>
                <li>Passes information forward through time</li>
            </ul>
            <h3>⚠️ Critical Problem</h3>
            <p><strong>Vanishing Gradients:</strong> As sequences get longer, gradients become exponentially small, preventing learning of long-term dependencies.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
            <h3>Architecture Components</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card">
            <h4>1. Embedding Layer</h4>
            <p>Converts word indices to dense vectors (128-dim)</p>
            
            <h4>2. Simple RNN Layer</h4>
            <p>64 hidden units<br>Recurrent connections</p>
            
            <h4>3. Dropout Layer</h4>
            <p>50% dropout rate<br>Prevents overfitting</p>
            
            <h4>4. Output Layer</h4>
            <p>Dense layer with sigmoid<br>Binary classification</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            <div class="success-card">
            <h2>LSTM Architecture</h2>
            <div class="code-block">
Model: LSTM
_________________________________________________
Layer (type)              Output Shape       Params
=================================================
embedding                 (None, 200, 128)   1,280,000
lstm                      (None, 64)         49,408
dropout (0.5)             (None, 64)         0
dense + sigmoid           (None, 1)          65
=================================================
Total params: 1,329,473
Trainable params: 1,329,473
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
            <h3>How LSTM Works</h3>
            <ul>
                <li><strong>Cell State:</strong> Highway for information flow</li>
                <li><strong>Forget Gate:</strong> Decides what to remove from memory</li>
                <li><strong>Input Gate:</strong> Decides what new information to store</li>
                <li><strong>Output Gate:</strong> Decides what to output</li>
            </ul>
            <h3>✅ Solves Vanishing Gradient</h3>
            <p>Gating mechanism allows gradients to flow unchanged through many time steps, enabling learning of long-term dependencies.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
            <h3>LSTM Gates</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="success-card">
            <h4>🚪 Forget Gate</h4>
            <p>f(t) = σ(W_f × [h(t-1), x(t)])</p>
            <p>Removes irrelevant information</p>
            
            <h4>📥 Input Gate</h4>
            <p>i(t) = σ(W_i × [h(t-1), x(t)])</p>
            <p>Adds new information</p>
            
            <h4>📤 Output Gate</h4>
            <p>o(t) = σ(W_o × [h(t-1), x(t)])</p>
            <p>Controls output</p>
            
            <h4>💾 Cell State</h4>
            <p>C(t) = f(t) × C(t-1) + i(t) × C̃(t)</p>
            <p>Long-term memory</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            <div class="success-card">
            <h2>GRU Architecture</h2>
            <div class="code-block">
Model: GRU
_________________________________________________
Layer (type)              Output Shape       Params
=================================================
embedding                 (None, 200, 128)   1,280,000
gru                       (None, 64)         37,056
dropout (0.5)             (None, 64)         0
dense + sigmoid           (None, 1)          65
=================================================
Total params: 1,317,121
Trainable params: 1,317,121
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
            <h3>How GRU Works</h3>
            <ul>
                <li><strong>Reset Gate:</strong> Decides how much past info to forget</li>
                <li><strong>Update Gate:</strong> Decides how much new info to add</li>
                <li>Combines cell state and hidden state into one</li>
            </ul>
            <h3>✅ Benefits Over LSTM</h3>
            <ul>
                <li>Fewer parameters (37K vs 49K in recurrent layer)</li>
                <li>Faster training and inference</li>
                <li>Similar performance to LSTM</li>
                <li>Simpler architecture, easier to tune</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
            <h3>GRU Gates</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="success-card">
            <h4>🔄 Reset Gate</h4>
            <p>r(t) = σ(W_r × [h(t-1), x(t)])</p>
            <p>Controls past memory influence</p>
            
            <h4>⬆️ Update Gate</h4>
            <p>z(t) = σ(W_z × [h(t-1), x(t)])</p>
            <p>Balances old vs new info</p>
            
            <h4>🎯 Hidden State</h4>
            <p>h(t) = (1-z(t)) × h(t-1) + z(t) × h̃(t)</p>
            <p>Combined memory</p>
            
            <h4>⚡ Efficiency</h4>
            <p>2 gates vs 3 in LSTM</p>
            <p>25% fewer parameters</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Common Configuration</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3>10,000</h3>
            <p>Vocabulary Size</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h3>200</h3>
            <p>Sequence Length</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h3>128</h3>
            <p>Embedding Dim</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-box">
            <h3>64</h3>
            <p>Hidden Units</p>
        </div>
        """, unsafe_allow_html=True)

# PAGE 4: TRAINING RESULTS
elif page == "📊 Training Results":
    st.markdown('<div class="section-header">Training Performance Analysis</div>', unsafe_allow_html=True)
    
    model_select = st.selectbox("🎯 Select Model to Analyze", ["RNN", "LSTM", "GRU"], 
                                 key="model_selector")
    
    df = training_data[model_select]
    
    # Create subplot with accuracy and loss
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Accuracy Over Epochs', 'Loss Over Epochs'),
        horizontal_spacing=0.12
    )
    
    # Accuracy subplot
    fig.add_trace(
        go.Scatter(x=df['Epoch'], y=df['Train Accuracy'],
                   mode='lines+markers', name='Train Accuracy',
                   line=dict(color='#00d4ff', width=3),
                   marker=dict(size=10)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['Epoch'], y=df['Val Accuracy'],
                   mode='lines+markers', name='Val Accuracy',
                   line=dict(color='#00ff88', width=3),
                   marker=dict(size=10)),
        row=1, col=1
    )
    
    # Loss subplot
    fig.add_trace(
        go.Scatter(x=df['Epoch'], y=df['Train Loss'],
                   mode='lines+markers', name='Train Loss',
                   line=dict(color='#ff6b6b', width=3),
                   marker=dict(size=10)),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=df['Epoch'], y=df['Val Loss'],
                   mode='lines+markers', name='Val Loss',
                   line=dict(color='#ffd93d', width=3),
                   marker=dict(size=10)),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Epoch", gridcolor='rgba(255,255,255,0.1)', row=1, col=1)
    fig.update_xaxes(title_text="Epoch", gridcolor='rgba(255,255,255,0.1)', row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", gridcolor='rgba(255,255,255,0.1)', row=1, col=1)
    fig.update_yaxes(title_text="Loss", gridcolor='rgba(255,255,255,0.1)', row=1, col=2)
    
    fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.2)',
        font=dict(color='white', size=14),
        hovermode='x unified',
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0.5)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics Display
    st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Final Train Acc</h4>
            <div class="stats-number">{df['Train Accuracy'].iloc[-1]:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Final Val Acc</h4>
            <div class="stats-number">{df['Val Accuracy'].iloc[-1]:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Final Train Loss</h4>
            <div class="stats-number">{df['Train Loss'].iloc[-1]:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Final Val Loss</h4>
            <div class="stats-number">{df['Val Loss'].iloc[-1]:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Table
    st.markdown("### 📋 Detailed Epoch-by-Epoch Breakdown")
    
    styled_df = df.style.background_gradient(cmap='Blues', subset=['Train Accuracy', 'Val Accuracy'])\
                        .background_gradient(cmap='Reds_r', subset=['Train Loss', 'Val Loss'])\
                        .format({'Train Accuracy': '{:.2f}%', 'Val Accuracy': '{:.2f}%',
                                'Train Loss': '{:.4f}', 'Val Loss': '{:.4f}'})
    
    st.dataframe(styled_df, use_container_width=True, height=250)
    
    # Model-specific observations
    st.markdown('<div class="section-header">Key Observations</div>', unsafe_allow_html=True)
    
    if model_select == "RNN":
        st.markdown("""
        <div class="error-card">
        <h2>❌ RNN Failed to Learn</h2>
        <h3>Critical Issues:</h3>
        <ul>
            <li><strong>Stagnant Accuracy:</strong> Remains around 50% throughout training (equivalent to random guessing)</li>
            <li><strong>No Improvement:</strong> Loss barely decreases, indicating the model cannot learn meaningful patterns</li>
            <li><strong>Vanishing Gradients:</strong> Cannot propagate error signals through long sequences</li>
            <li><strong>Poor Generalization:</strong> Test accuracy of 49.62% confirms the model learned nothing</li>
        </ul>
        <h3>Why This Happened:</h3>
        <p>Movie reviews are relatively long sequences. Simple RNNs multiply gradients at each time step, causing them to 
        exponentially shrink (vanish) or grow (explode). This prevents the network from learning dependencies beyond a few words.</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif model_select == "LSTM":
        st.markdown("""
        <div class="success-card">
        <h2>✅ LSTM Successfully Learned</h2>
        <h3>Strong Performance:</h3>
        <ul>
            <li><strong>Steady Improvement:</strong> Training accuracy grows from 50% to 91.69% over 5 epochs</li>
            <li><strong>Good Generalization:</strong> Validation accuracy reaches 87.77%, close to training performance</li>
            <li><strong>Effective Learning:</strong> Loss decreases consistently from 0.69 to 0.23</li>
            <li><strong>Stable Training:</strong> No sudden drops or instability</li>
        </ul>
        <h3>Minor Concerns:</h3>
        <ul>
            <li><strong>Slight Overfitting:</strong> Gap between train (91.69%) and validation (87.77%) in final epoch</li>
            <li><strong>Slow Convergence:</strong> Takes 4-5 epochs to reach peak performance</li>
            <li><strong>Computational Cost:</strong> 376 seconds per epoch (slowest of all models)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    else:  # GRU
        st.markdown("""
        <div class="success-card">
        <h2>✅ GRU Achieved Best Results</h2>
        <h3>Outstanding Performance:</h3>
        <ul>
            <li><strong>Rapid Learning:</strong> Jumps from 50% to 87% validation accuracy by epoch 2</li>
            <li><strong>High Final Accuracy:</strong> Training accuracy reaches 97.81%, validation at 87.63%</li>
            <li><strong>Fast Convergence:</strong> Achieves strong results in just 2-3 epochs</li>
            <li><strong>Efficient Training:</strong> Only 292 seconds per epoch (22% faster than LSTM)</li>
        </ul>
        <h3>Trade-offs:</h3>
        <ul>
            <li><strong>Overfitting Present:</strong> Large gap between training (97.81%) and validation (87.63%)</li>
            <li><strong>Validation Fluctuation:</strong> Validation accuracy varies between 87-89% in later epochs</li>
            <li><strong>Early Stopping Recommended:</strong> Best to stop at epoch 3 to prevent overfitting</li>
        </ul>
        <h3>Why GRU Excels:</h3>
        <p>GRU's simpler architecture (2 gates vs LSTM's 3) makes it faster to train while maintaining similar expressive power. 
        The reset and update gates effectively control information flow without the complexity overhead.</p>
        </div>
        """, unsafe_allow_html=True)

# PAGE 5: MODEL COMPARISON
elif page == "⚖️ Model Comparison":
    st.markdown('<div class="section-header">Comprehensive Model Comparison</div>', unsafe_allow_html=True)
    
    # Test Accuracy Comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="error-card">
            <h2>RNN</h2>
            <div class="stats-number">49.62%</div>
            <p>Test Accuracy</p>
            <hr>
            <p>❌ Failed</p>
            <p>Random Guessing Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-card">
            <h2>LSTM</h2>
            <div class="stats-number">87.42%</div>
            <p>Test Accuracy</p>
            <hr>
            <p>✅ Success</p>
            <p>Stable & Reliable</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="success-card">
            <h2>GRU</h2>
            <div class="stats-number">87.72%</div>
            <p>Test Accuracy</p>
            <hr>
            <p>🏆 Best Overall</p>
            <p>Fast & Accurate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visual Comparison Charts
    st.markdown('<div class="section-header">Performance Visualizations</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy Comparison
        fig_acc = go.Figure(data=[
            go.Bar(name='Test Accuracy', 
                   x=test_results['Model'], 
                   y=test_results['Test Accuracy'],
                   text=test_results['Test Accuracy'].apply(lambda x: f'{x:.2f}%'),
                   textposition='outside',
                   marker=dict(color=['#ff4444', '#00ff88', '#00d4ff'],
                             line=dict(color='white', width=2)))
        ])
        
        fig_acc.update_layout(
            title="Test Accuracy Comparison",
            yaxis_title="Accuracy (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white', size=14),
            height=400,
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # Training Time Comparison
        fig_time = go.Figure(data=[
            go.Bar(name='Avg Time per Epoch', 
                   x=test_results['Model'], 
                   y=test_results['Avg Time per Epoch'],
                   text=test_results['Avg Time per Epoch'].apply(lambda x: f'{x}s'),
                   textposition='outside',
                   marker=dict(color=['#00ff88', '#ff4444', '#ffd93d'],
                             line=dict(color='white', width=2)))
        ])
        
        fig_time.update_layout(
            title="Training Time Comparison",
            yaxis_title="Time per Epoch (seconds)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white', size=14),
            height=400,
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # All Models Learning Curves
    st.markdown("### 📈 Learning Curves: All Models")
    
    fig_all = go.Figure()
    
    colors = {'RNN': '#ff4444', 'LSTM': '#00ff88', 'GRU': '#00d4ff'}
    
    for model_name in ['RNN', 'LSTM', 'GRU']:
        df = training_data[model_name]
        fig_all.add_trace(go.Scatter(
            x=df['Epoch'], 
            y=df['Val Accuracy'],
            mode='lines+markers',
            name=f'{model_name}',
            line=dict(width=4, color=colors[model_name]),
            marker=dict(size=12)
        ))
    
    fig_all.update_layout(
        xaxis_title='Epoch',
        yaxis_title='Validation Accuracy (%)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.2)',
        font=dict(color='white', size=16),
        height=500,
        hovermode='x unified',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)', font=dict(size=16))
    )
    
    st.plotly_chart(fig_all, use_container_width=True)
    
    # Detailed Comparison Table
    st.markdown('<div class="section-header">Detailed Metrics Table</div>', unsafe_allow_html=True)
    
    comparison_data = {
        'Metric': [
            'Test Accuracy',
            'Final Val Accuracy', 
            'Final Train Accuracy',
            'Best Val Accuracy',
            'Convergence Speed',
            'Avg Time per Epoch',
            'Total Training Time',
            'Parameters',
            'Overfitting Level'
        ],
        'RNN': [
            '49.62%', '50.45%', '51.47%', '50.45%',
            'No Convergence', '102s', '510s', '1.29M', 'None (no learning)'
        ],
        'LSTM': [
            '87.42%', '87.77%', '91.69%', '87.77%',
            'Slow (4-5 epochs)', '376s', '1880s', '1.33M', 'Low (3.92% gap)'
        ],
        'GRU': [
            '87.72%', '87.63%', '97.81%', '89.05%',
            'Fast (2-3 epochs)', '292s', '1460s', '1.32M', 'Moderate (10.18% gap)'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df.style.set_properties(**{
            'background-color': 'rgba(255,255,255,0.05)',
            'color': 'white',
            'border-color': 'rgba(0,212,255,0.2)'
        }),
        use_container_width=True,
        height=400
    )
    
    # Key Insights
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>🎯 Accuracy Winner: GRU</h3>
        <ul>
            <li>Highest test accuracy: <strong>87.72%</strong></li>
            <li>0.3% better than LSTM</li>
            <li>37.8% better than RNN</li>
        </ul>
        
        <h3>⚡ Speed Winner: RNN</h3>
        <ul>
            <li>Fastest training: <strong>102s/epoch</strong></li>
            <li>But accuracy is unusable at 49.62%</li>
            <li>Speed means nothing without learning</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-card">
        <h3>🏆 Best Balance: GRU</h3>
        <ul>
            <li><strong>22% faster</strong> than LSTM (292s vs 376s)</li>
            <li><strong>0.3% more accurate</strong> than LSTM</li>
            <li><strong>Converges faster:</strong> 2-3 epochs vs 4-5</li>
            <li><strong>Fewer parameters:</strong> More efficient</li>
        </ul>
        
        <h3>💡 LSTM Advantage</h3>
        <ul>
            <li>More stable training curve</li>
            <li>Lower overfitting (3.92% vs 10.18%)</li>
            <li>Better for production if stability is critical</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# PAGE 6: CONCLUSIONS
elif page == "🎯 Conclusions":
    st.markdown('<div class="section-header">Final Verdict</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-card" style="text-align: center;">
    <h1>🏆 Winner: GRU (Gated Recurrent Unit)</h1>
    <div class="stats-number">87.72%</div>
    <h3>Test Accuracy</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Why GRU is the Best Choice</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-card">
        <h3>✅ Advantages</h3>
        <ol>
            <li><strong>Highest Accuracy:</strong> 87.72% test accuracy, beating LSTM by 0.3%</li>
            <li><strong>Faster Training:</strong> 292s per epoch vs 376s for LSTM (22% faster)</li>
            <li><strong>Quick Convergence:</strong> Reaches peak performance by epoch 2-3</li>
            <li><strong>Efficient Architecture:</strong> Fewer parameters than LSTM (37K vs 49K in recurrent layer)</li>
            <li><strong>Lower Complexity:</strong> 2 gates instead of 3, easier to understand and tune</li>
            <li><strong>Production Ready:</strong> Fast inference time for real-world deployment</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-card">
        <h3>⚠️ Considerations</h3>
        <ul>
            <li><strong>Overfitting:</strong> Shows 10.18% gap between train and validation in final epoch</li>
            <li><strong>Solution:</strong> Use early stopping at epoch 3 when validation accuracy peaks at 89.05%</li>
            <li><strong>Validation Volatility:</strong> Accuracy fluctuates slightly in later epochs</li>
            <li><strong>Recommendation:</strong> Monitor validation metrics and stop when they plateau</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Model-by-Model Summary</div>', unsafe_allow_html=True)
    
    # RNN Summary
    st.markdown("""
    <div class="error-card">
    <h2>❌ Simple RNN: Complete Failure</h2>
    <h3>Test Accuracy: 49.62%</h3>
    <p><strong>Why it failed:</strong></p>
    <ul>
        <li><strong>Vanishing Gradient Problem:</strong> Gradients exponentially shrink through time steps</li>
        <li><strong>Cannot Learn Long Dependencies:</strong> Movie reviews require understanding context across many words</li>
        <li><strong>Stuck at Random Guess:</strong> 50% accuracy is equivalent to flipping a coin</li>
    </ul>
    <p><strong>Verdict:</strong> Never use simple RNN for sentiment analysis or any long-sequence task.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # LSTM Summary
    st.markdown("""
    <div class="success-card">
    <h2>✅ LSTM: Solid and Reliable</h2>
    <h3>Test Accuracy: 87.42%</h3>
    <p><strong>Strengths:</strong></p>
    <ul>
        <li><strong>Stable Training:</strong> Consistent improvement across epochs</li>
        <li><strong>Good Generalization:</strong> Only 3.92% gap between train and validation</li>
        <li><strong>Industry Standard:</strong> Proven architecture for NLP tasks</li>
        <li><strong>Reliable:</strong> Predictable behavior, good for production</li>
    </ul>
    <p><strong>Weaknesses:</strong></p>
    <ul>
        <li><strong>Slow Training:</strong> 376s per epoch (slowest model)</li>
        <li><strong>Complex:</strong> 3 gates make it harder to interpret</li>
        <li><strong>More Parameters:</strong> Higher computational cost</li>
    </ul>
    <p><strong>Verdict:</strong> Use when stability and reliability are more important than speed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # GRU Summary
    st.markdown("""
    <div class="success-card">
    <h2>🏆 GRU: Best Overall Performance</h2>
    <h3>Test Accuracy: 87.72%</h3>
    <p><strong>Strengths:</strong></p>
    <ul>
        <li><strong>Best Accuracy:</strong> Highest test performance at 87.72%</li>
        <li><strong>Fast Training:</strong> 22% faster than LSTM</li>
        <li><strong>Quick Convergence:</strong> Achieves good results in just 2-3 epochs</li>
        <li><strong>Efficient:</strong> Fewer parameters, lower memory footprint</li>
        <li><strong>Practical:</strong> Best choice for deployment scenarios</li>
    </ul>
    <p><strong>Weaknesses:</strong></p>
    <ul>
        <li><strong>Overfitting Tendency:</strong> Shows larger train-val gap in later epochs</li>
        <li><strong>Requires Monitoring:</strong> Need early stopping to prevent overfitting</li>
    </ul>
    <p><strong>Verdict:</strong> Best default choice for most sentiment analysis and text classification tasks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Technical Deep Dive</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔴 Why RNN Failed", "🟢 Why LSTM/GRU Succeed", "🔵 GRU vs LSTM"])
    
    with tab1:
        st.markdown("""
        <div class="info-card">
        <h3>The Vanishing Gradient Problem</h3>
        <p>During backpropagation through time (BPTT), gradients are multiplied at each time step:</p>
        <div class="code-block">
        ∂L/∂h₀ = ∂L/∂hₜ × ∂hₜ/∂hₜ₋₁ × ∂hₜ₋₁/∂hₜ₋₂ × ... × ∂h₁/∂h₀
        </div>
        <p>For sequences of length T, this involves T multiplications. If each gradient term is less than 1, 
        the product becomes exponentially small (vanishes). If greater than 1, it explodes.</p>
        
        <h3>Impact on Learning</h3>
        <ul>
            <li>Early layers receive almost zero gradient signal</li>
            <li>Network cannot learn long-term dependencies</li>
            <li>Only recent words influence predictions</li>
            <li>For movie reviews spanning 200 words, RNN only "sees" last few words</li>
        </ul>
        
        <h3>Example</h3>
        <p>Review: "The movie started great but the ending was terrible and disappointing"</p>
        <p>RNN only remembers: "...terrible and disappointing" → Predicts negative ✓</p>
        <p>But if review is: "Despite some terrible acting, the plot was great and heartwarming"</p>
        <p>RNN only remembers: "...great and heartwarming" but misses "terrible" context → Incorrect</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="info-card">
        <h3>The Gating Mechanism Solution</h3>
        <p>Both LSTM and GRU use <strong>gates</strong> to control information flow:</p>
        
        <h3>LSTM Approach (3 Gates)</h3>
        <ol>
            <li><strong>Forget Gate:</strong> Decides what information to discard from cell state</li>
            <li><strong>Input Gate:</strong> Decides what new information to add to cell state</li>
            <li><strong>Output Gate:</strong> Decides what to output based on cell state</li>
        </ol>
        <div class="code-block">
        Cell State acts as a "memory highway":
        C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ C̃(t)
        
        Gradients flow through this highway without multiplication,
        solving the vanishing gradient problem!
        </div>
        
        <h3>GRU Approach (2 Gates)</h3>
        <ol>
            <li><strong>Reset Gate:</strong> Decides how much past information to forget</li>
            <li><strong>Update Gate:</strong> Decides how much new vs old information to use</li>
        </ol>
        <div class="code-block">
        Combines cell state and hidden state:
        h(t) = (1-z(t)) ⊙ h(t-1) + z(t) ⊙ h̃(t)
        
        Simpler than LSTM but equally effective!
        </div>
        
        <h3>Key Advantage</h3>
        <p>Gates can learn to remain open, allowing gradients to flow unchanged through many time steps. 
        This enables learning of long-range dependencies that are crucial for understanding full review context.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="info-card">
        <h3>GRU vs LSTM: The Key Differences</h3>
        
        <h3>Architecture Comparison</h3>
        <table style="width:100%; color:white;">
        <tr style="background-color:rgba(0,212,255,0.2);">
            <th>Aspect</th>
            <th>LSTM</th>
            <th>GRU</th>
        </tr>
        <tr>
            <td><strong>Number of Gates</strong></td>
            <td>3 (Forget, Input, Output)</td>
            <td>2 (Reset, Update)</td>
        </tr>
        <tr style="background-color:rgba(255,255,255,0.05);">
            <td><strong>Parameters (64 units)</strong></td>
            <td>49,408</td>
            <td>37,056 (25% fewer)</td>
        </tr>
        <tr>
            <td><strong>Cell State</strong></td>
            <td>Separate from hidden state</td>
            <td>Merged with hidden state</td>
        </tr>
        <tr style="background-color:rgba(255,255,255,0.05);">
            <td><strong>Training Speed</strong></td>
            <td>376s per epoch</td>
            <td>292s per epoch (22% faster)</td>
        </tr>
        <tr>
            <td><strong>Memory Usage</strong></td>
            <td>Higher</td>
            <td>Lower</td>
        </tr>
        </table>
        
        <h3>When to Choose Which?</h3>
        
        <h4>Choose GRU when:</h4>
        <ul>
            <li>You need faster training and inference</li>
            <li>You have limited computational resources</li>
            <li>Your sequences are moderately long (like movie reviews)</li>
            <li>You want to iterate quickly during development</li>
            <li>You're deploying to production and need efficiency</li>
        </ul>
        
        <h4>Choose LSTM when:</h4>
        <ul>
            <li>You need maximum stability in training</li>
            <li>You're working with very long sequences (1000+ tokens)</li>
            <li>Interpretability of gates is important</li>
            <li>You have ample computational resources</li>
            <li>Prior research in your domain used LSTM successfully</li>
        </ul>
        
        <h3>Our Recommendation: GRU</h3>
        <p>For sentiment analysis on movie reviews, GRU is the clear winner because:</p>
        <ul>
            <li>Sequences are manageable length (200 tokens)</li>
            <li>Speed matters for practical deployment</li>
            <li>0.3% accuracy gain over LSTM</li>
            <li>Lower resource requirements</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Project Summary</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
    <h3>🎬 Complete Project Overview</h3>
    
    <h4>Dataset</h4>
    <p>Kaggle Movie Reviews Dataset with binary sentiment labels (positive/negative)</p>
    
    <h4>Preprocessing Pipeline</h4>
    <ul>
        <li>Text cleaning: lowercase, remove HTML/URLs/numbers/punctuation</li>
        <li>Stopword removal for noise reduction</li>
        <li>Tokenization: vocabulary of 10,000 words</li>
        <li>Padding: fixed sequence length of 200 tokens</li>
        <li>80-20 train-test split with stratification</li>
    </ul>
    
    <h4>Models Tested</h4>
    <ol>
        <li><strong>Simple RNN:</strong> Baseline model with vanishing gradient issues</li>
        <li><strong>LSTM:</strong> Advanced architecture with 3-gate mechanism</li>
        <li><strong>GRU:</strong> Efficient architecture with 2-gate mechanism</li>
    </ol>
    
    <h4>Results Summary</h4>
    <table style="width:100%; color:white; margin-top:1rem;">
    <tr style="background-color:rgba(0,212,255,0.2);">
        <th>Model</th>
        <th>Test Accuracy</th>
        <th>Time/Epoch</th>
        <th>Status</th>
    </tr>
    <tr>
        <td>RNN</td>
        <td>49.62%</td>
        <td>102s</td>
        <td>❌ Failed</td>
    </tr>
    <tr style="background-color:rgba(255,255,255,0.05);">
        <td>LSTM</td>
        <td>87.42%</td>
        <td>376s</td>
        <td>✅ Success</td>
    </tr>
    <tr>
        <td>GRU</td>
        <td>87.72%</td>
        <td>292s</td>
        <td>🏆 Best</td>
    </tr>
    </table>
    
    <h4>Key Findings</h4>
    <ul>
        <li>Simple RNNs are inadequate for sentiment analysis due to vanishing gradients</li>
        <li>LSTM and GRU both achieve high accuracy by solving the gradient problem</li>
        <li>GRU provides the best balance of accuracy, speed, and efficiency</li>
        <li>Gating mechanisms are essential for learning long-term dependencies in text</li>
    </ul>
    
    <h4>Final Recommendation</h4>
    <p><strong>Deploy GRU model</strong> for production sentiment analysis with:</p>
    <ul>
        <li>Early stopping at epoch 3 to prevent overfitting</li>
        <li>Regular monitoring of validation metrics</li>
        <li>Expected real-world accuracy: ~87-89%</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Future Improvements
    st.markdown('<div class="section-header">Potential Improvements</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>📈 Model Enhancements</h3>
        <ul>
            <li><strong>Bidirectional GRU:</strong> Process sequences in both directions for better context</li>
            <li><strong>Attention Mechanism:</strong> Focus on important words in the review</li>
            <li><strong>Ensemble Methods:</strong> Combine multiple GRU models for higher accuracy</li>
            <li><strong>Pre-trained Embeddings:</strong> Use GloVe or Word2Vec instead of random initialization</li>
            <li><strong>Deeper Networks:</strong> Stack multiple GRU layers for more complex patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>🔧 Training Optimizations</h3>
        <ul>
            <li><strong>Learning Rate Scheduling:</strong> Gradually decrease learning rate</li>
            <li><strong>Early Stopping:</strong> Stop at epoch 3 to prevent overfitting</li>
            <li><strong>Regularization:</strong> Increase dropout or add L2 regularization</li>
            <li><strong>Data Augmentation:</strong> Synonym replacement, back-translation</li>
            <li><strong>Hyperparameter Tuning:</strong> Grid search for optimal parameters</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Modern Alternatives
    st.markdown('<div class="section-header">Modern Alternatives</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-card">
    <h3>🚀 Transformer-based Models</h3>
    <p>While GRU achieved 87.72% accuracy, modern transformer-based models can achieve even higher performance:</p>
    
    <h4>BERT (Bidirectional Encoder Representations from Transformers)</h4>
    <ul>
        <li>Expected accuracy: 92-95% on sentiment analysis</li>
        <li>Pre-trained on massive text corpora</li>
        <li>Better understanding of context and nuance</li>
        <li>Trade-off: Much slower and requires more computational resources</li>
    </ul>
    
    <h4>When to Use Transformers vs GRU</h4>
    <table style="width:100%; color:white; margin-top:1rem;">
    <tr style="background-color:rgba(0,212,255,0.2);">
        <th>Scenario</th>
        <th>Recommended Model</th>
    </tr>
    <tr>
        <td>Production with limited resources</td>
        <td>GRU (fast, efficient, good accuracy)</td>
    </tr>
    <tr style="background-color:rgba(255,255,255,0.05);">
        <td>Maximum accuracy required</td>
        <td>BERT or similar transformers</td>
    </tr>
    <tr>
        <td>Real-time inference needed</td>
        <td>GRU (low latency)</td>
    </tr>
    <tr style="background-color:rgba(255,255,255,0.05);">
        <td>Learning/educational project</td>
        <td>GRU (easier to understand)</td>
    </tr>
    <tr>
        <td>Large labeled dataset available</td>
        <td>Either (transformers better with more data)</td>
    </tr>
    </table>
    
    <p><strong>Note:</strong> For this project, GRU provides the best balance of performance and practicality.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Conclusion
    st.markdown('<div class="section-header">Final Thoughts</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-card" style="text-align: center; padding: 3rem;">
    <h2>🎯 Project Success</h2>
    <p style="font-size: 1.2rem; margin: 2rem 0;">
    This project successfully demonstrated that sequential neural networks, specifically GRU, 
    can effectively learn sentiment from movie reviews with 87.72% accuracy.
    </p>
    <p style="font-size: 1.1rem;">
    The comparison clearly showed why simple RNNs fail and how gating mechanisms in 
    LSTM and GRU solve the vanishing gradient problem, enabling deep learning on textual data.
    </p>
    <div style="margin-top: 2rem;">
        <div class="stats-number" style="font-size: 4rem;">GRU</div>
        <p style="font-size: 1.3rem; color: #00d4ff;">The Clear Winner</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Project:** Sequential Models for Sentiment Analysis")
with col2:
    st.markdown("**Models:** RNN | LSTM | GRU")
with col3:
    st.markdown("**Best Model:** GRU (87.72%)")