# CompI Phase 2.B: Data/Logic Input to Image Generation

## 🚀 Overview

Phase 2.B transforms structured data and mathematical formulas into stunning AI-generated art. This phase combines data analysis, pattern recognition, and poetic interpretation to create unique visual experiences that reflect the essence of your data.

## ✨ Key Features

### 📊 Data Processing
- **CSV Data Analysis**: Upload spreadsheets, time series, measurements, or any numeric data
- **Mathematical Formula Evaluation**: Enter Python/NumPy expressions for mathematical art
- **Pattern Recognition**: Automatic detection of trends, correlations, and seasonality
- **Statistical Analysis**: Comprehensive data profiling and feature extraction

### 🎨 Artistic Integration
- **Poetic Text Generation**: Convert data patterns into descriptive, artistic language
- **Data Visualization**: Create beautiful charts and plots from your data
- **Prompt Enhancement**: Intelligently merge data insights with your creative prompts
- **Visual Conditioning**: Use data visualizations to inspire AI art generation

### 🔧 Technical Capabilities
- **Safe Formula Execution**: Secure evaluation of mathematical expressions
- **Batch Processing**: Handle multiple datasets or formulas simultaneously
- **Comprehensive Metadata**: Detailed logging of all generation parameters
- **Flexible Output**: Save both generated art and data visualizations

## 🛠️ Installation & Setup

### Prerequisites
Ensure you have the base CompI environment set up with all dependencies from `requirements.txt`.

### Additional Dependencies
Phase 2.B uses the existing CompI dependencies, specifically:
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Mathematical operations
- `matplotlib>=3.7.0` - Data visualization
- `seaborn>=0.12.0` - Statistical plotting

## 🎯 Quick Start

### 1. Launch the Streamlit Interface

```bash
# Navigate to your CompI project directory
cd "C:\Users\Aksharajsinh\Documents\augment-projects\Project CompI"

# Run the Phase 2.B interface
streamlit run src/ui/compi_phase2b_streamlit_ui.py
```

### 2. Using CSV Data

1. **Upload your CSV file** containing numeric data
2. **Enter your creative prompt** (e.g., "A flowing river of data")
3. **Set style and mood** (e.g., "abstract digital art", "serene and flowing")
4. **Click Generate** and watch your data transform into art!

### 3. Using Mathematical Formulas

1. **Enter a mathematical formula** using Python/NumPy syntax
2. **Combine with your prompt** for artistic interpretation
3. **Generate unique mathematical art** based on your equations

## 📚 Examples

### CSV Data Examples

#### Time Series Data
```csv
date,temperature,humidity,pressure
2024-01-01,22.5,65,1013.2
2024-01-02,23.1,62,1015.8
2024-01-03,21.8,68,1012.4
...
```

**Prompt**: "Weather patterns dancing across the sky"
**Style**: "impressionist painting"
**Result**: Art inspired by temperature fluctuations and atmospheric pressure

#### Financial Data
```csv
date,price,volume,volatility
2024-01-01,100.5,1000000,0.15
2024-01-02,102.3,1200000,0.18
2024-01-03,99.8,900000,0.22
...
```

**Prompt**: "The rhythm of market forces"
**Style**: "geometric abstract"
**Result**: Visual representation of market dynamics

### Mathematical Formula Examples

#### Sine Wave with Decay
```python
np.sin(np.linspace(0, 4*np.pi, 100)) * np.exp(-np.linspace(0, 1, 100))
```
**Prompt**: "Fading echoes in a digital realm"
**Result**: Art representing diminishing oscillations

#### Spiral Pattern
```python
t = np.linspace(0, 4*np.pi, 200)
np.sin(t) * t
```
**Prompt**: "The golden ratio in nature"
**Result**: Spiral-inspired organic art

#### Complex Harmonic
```python
x = np.linspace(0, 6*np.pi, 300)
np.sin(x) + 0.5*np.cos(3*x) + 0.25*np.sin(5*x)
```
**Prompt**: "Musical harmonies visualized"
**Result**: Multi-layered wave patterns

## 🎨 Creative Workflow

### 1. Data Preparation
- **Clean your data**: Remove or handle missing values
- **Choose meaningful columns**: Focus on numeric data that tells a story
- **Consider time series**: Temporal data often creates compelling patterns

### 2. Prompt Engineering
- **Start with your data story**: What does your data represent?
- **Add artistic style**: Choose styles that complement your data's nature
- **Set the mood**: Match the emotional tone to your data's characteristics

### 3. Style Recommendations

| Data Type | Recommended Styles | Mood Suggestions |
|-----------|-------------------|------------------|
| Time Series | flowing, organic, wave-like | rhythmic, temporal, evolving |
| Statistical | geometric, structured, minimal | analytical, precise, clean |
| Financial | dynamic, angular, sharp | energetic, volatile, intense |
| Scientific | technical, detailed, precise | methodical, systematic, clear |
| Random/Chaotic | abstract, expressionist, wild | unpredictable, chaotic, free |

## 🔧 Advanced Usage

### Programmatic Access

```python
from src.generators.compi_phase2b_data_to_image import CompIPhase2BDataToImage

# Initialize generator
generator = CompIPhase2BDataToImage()

# Generate from CSV
results = generator.generate_image(
    text_prompt="Data flowing like water",
    style="fluid abstract",
    mood="serene, continuous",
    csv_path="path/to/your/data.csv",
    num_images=2
)

# Generate from formula
results = generator.generate_image(
    text_prompt="Mathematical harmony",
    style="geometric precision",
    mood="balanced, rhythmic",
    formula="np.sin(np.linspace(0, 4*np.pi, 100))",
    num_images=1
)
```

### Batch Processing

```python
# Process multiple CSV files
results = generator.batch_process_csv_files(
    csv_directory="data/experiments/",
    text_prompt="Scientific visualization",
    style="technical illustration",
    mood="precise, analytical"
)

# Process multiple formulas
formulas = [
    "np.sin(x)",
    "np.cos(x)",
    "np.tan(x/2)"
]
results = generator.batch_process_formulas(
    formulas=formulas,
    text_prompt="Trigonometric art",
    style="mathematical beauty"
)
```

## 📊 Understanding Data Features

Phase 2.B analyzes your data and extracts several key features:

### Statistical Features
- **Means, Medians, Standard Deviations**: Basic statistical measures
- **Ranges and Distributions**: Data spread and shape
- **Trends**: Increasing, decreasing, stable, or volatile patterns

### Pattern Features
- **Correlations**: Relationships between different data columns
- **Seasonality**: Repeating patterns in time series data
- **Complexity Score**: Measure of data intricacy (0-1)
- **Variability Score**: Measure of data diversity (0-1)
- **Pattern Strength**: Measure of detectable patterns (0-1)

### Poetic Interpretation
The system converts these features into artistic language:
- **Trend descriptions**: "ascending", "flowing", "turbulent"
- **Pattern adjectives**: "intricate", "harmonious", "dynamic"
- **Artistic metaphors**: "like brushstrokes on canvas", "dancing with precision"

## 🎯 Tips for Best Results

### Data Tips
1. **Quality over quantity**: Clean, meaningful data works better than large messy datasets
2. **Numeric focus**: Ensure your CSV has numeric columns for analysis
3. **Reasonable size**: Keep datasets under 10,000 rows for faster processing
4. **Meaningful names**: Use descriptive column names for better interpretation

### Formula Tips
1. **Use NumPy functions**: Leverage `np.sin`, `np.cos`, `np.exp`, etc.
2. **Define ranges**: Use `np.linspace()` to create smooth curves
3. **Experiment with complexity**: Combine multiple functions for richer patterns
4. **Consider scale**: Ensure your formula produces reasonable numeric ranges

### Prompt Tips
1. **Be descriptive**: Rich prompts lead to more interesting results
2. **Match your data**: Align artistic style with data characteristics
3. **Experiment**: Try different style/mood combinations
4. **Use the preview**: Check the enhanced prompt before generating

## 🔍 Troubleshooting

### Common Issues

**"Error analyzing data"**
- Check that your CSV has numeric columns
- Ensure the file is properly formatted
- Try with a smaller dataset first

**"Invalid formula"**
- Use only safe mathematical functions
- Check your NumPy syntax
- Ensure parentheses are balanced

**"Generation failed"**
- Check your GPU memory if using CUDA
- Try reducing the number of inference steps
- Ensure your prompt isn't too long

### Performance Optimization
- Use GPU acceleration when available
- Reduce image dimensions for faster generation
- Process smaller datasets for quicker analysis
- Use fewer inference steps for rapid prototyping

## 🚀 Next Steps

After mastering Phase 2.B, consider:
1. **Combining with Phase 2.A**: Use audio + data for multimodal art
2. **Creating data stories**: Build narratives around your visualizations
3. **Exploring advanced formulas**: Try complex mathematical expressions
4. **Building datasets**: Create custom data for specific artistic goals

---

**Ready to transform your data into art?** Launch the Streamlit interface and start creating! 🎨📊✨
