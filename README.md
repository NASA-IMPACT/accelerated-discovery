# 🤖 **Accelerated Discovery:**  
**Empowering Research Scientists with AI Automation**  


---
## 📌 **About**  

---
##  **Technical Overview**  
---

## 🚀 **Installation Procedure**  

### **1️⃣ Clone the Repository**
```sh
git clone git@github.com:NASA-IMPACT/accelerated-discovery.git
cd accelerated-discovery
```

### **2️⃣ Set Up the Conda Environment**
```sh
conda create -n ad python=3.11
conda activate ad
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```


### **4️⃣ Set Environment Variables**
Copy the `.env` template and configure the **AGENT_PERSISTANCE_PATH**:

```sh
cp .env_copy .env
```
Edit `.env` and set:

```
SERPER_API_KEY 
WATSONX_API_KEY
WATSONX_PROJECT_ID
```
NOTE: because of a bug, please copy the same file .env under accelerated_discovery/tools/


### 🖥 **Run Litarature Review Flow**
run single workflow 
```sh
python accelerated_discovery/frontend/literature_review.py' 
```


use streamlit interface
```sh
streamlit run accelerated_discovery/frontend/literature_review.py
```


---

## 🤝 **Contributing**
🔹 **Contributors are welcome!**  

- **First step:** Clone the repo and **run it**.  
- **Report any issues** and assign them to **@gliozzo**.  
- **To add new features**, create an issue and assign it to yourself.  
- PRs **can be merged without review** in the initial phase but may be reverted if problematic.  

---

## 📬 **Contact**
💡 Have questions or suggestions? Reach out:  
📧 Email: [gliozzo@us.ibm.com](mailto:gliozzo@us.ibm.com)

---

### **🌟 Star this repository if you find it useful!** ⭐  

---








