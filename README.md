# 📄 Official Implementation

**Official implementation of the paper:**  
**_Rewriting the Budget: A General Framework for Black-Box Attacks Under Cost Asymmetry_**

This repository provides the codebase to reproduce experiments presented in the paper, including support for several state-of-the-art black-box adversarial attacks under asymmetric cost settings.

---

## 🔧 Running Attacks

You can run different attacks by specifying the desired configuration through `main.py`. The key parameters include the attack name, total query budget, cost settings, and device selection.

### ✅ Basic Usage

`python main.py --attack <ATTACK_NAME> --device <DEVICE> --total-cost <COST> [OPTIONS...]`

---

## 🚀 Supported Attacks & Examples

### 🔹 SURFREE

`python main.py --attack SURFREE --device "cuda:0" --total-cost 10000 --dimension-reduction-mode "Full" --dimension-reduction-factor 4.0 --search-cost 1 --query-cost 2`

### 🔹 OPT

`python main.py --attack OPT --device "cuda:0" --total-cost 15000 --search-cost 1 --query-cost 5`

### 🔹 HSJA

`python main.py --attack HSJA --device "cuda:0" --total-cost 10000 --initial-gradient-queries 30 --search-cost 1 --query-cost 2`

### 🔹 GeoDA

`python main.py --attack GEODA --device "cuda:0" --total-cost 150000 --initial-gradient-queries 30 --dimension-reduction-mode "Full" --dimension-reduction-factor 4.0 --search-cost 1 --query-cost 100`

### 🔹 CGBA

`python main.py --attack CGBA --device "cuda:0" --total-cost 10000 --initial-gradient-queries 30 --dimension-reduction-mode "Full" --dimension-reduction-factor 4.0 --search-cost 1 --query-cost 2`

---

## ⚙️ Key Arguments

- `--attack`: Attack name (`SURFREE`, `OPT`, `HSJA`, `GEODA`, `CGBA`)
- `--device`: Device to run on (e.g., `cuda:0`, `cpu`)
- `--total-cost`: Total allowed query budget
- `--search-cost`: Cost of a single query used during **Asymmetric Search**
- `--query-cost`: Cost of the final attack query; defines **asymmetry ratio** \( c^\star \)
- `--initial-gradient-queries`: Number of initial queries for gradient estimation (used in gradient-based attacks)
- `--dimension-reduction-mode`, `--dimension-reduction-factor`: Options to reduce the attack dimension (e.g., `"Full"`, `4.0`)
- `--save-trajectories`: Save intermediate perturbed images for trajectory visualization

---

## 📊 Visualizations

### Plot Attack Results
Generate line plots comparing attack performance:

```bash
python3 plot.py
```

This creates `attack_results_plot.png` and `attack_results_plot.pdf` showing median L₂ perturbation vs cost for different attacks.

### Visualize Attack Trajectories
Generate grid visualizations showing how images evolve during attacks:

```bash
python3 visualize_trajectories.py --attack HSJA --total-cost 10000 --query-cost 1.0
```

For comparing multiple attacks:
```bash
python3 visualize_trajectories.py --compare --attacks HSJA CGBA SURFREE --total-cost 10000
```

**Note:** Run attacks with `--save-trajectories` flag first to generate trajectory data.

---

## 🖼️ Using Custom Images

You can use your own images instead of ImageNet validation set:

1. **Setup custom images:**
   ```bash
   python3 setup_custom_images.py
   ```
   Place your images in the `custom_images/` folder (supports .jpg, .jpeg, .png)

2. **Run attacks with trajectories:**
   ```bash
   python3 main.py --attack HSJA --save-trajectories --device cpu --total-cost 1000 --image-start 1 --image-end 6
   ```

3. **Visualize the results:**
   ```bash
   python3 visualize_trajectories.py --attack HSJA --total-cost 1000 --query-cost 1.0
   ```

See `README_CUSTOM_IMAGES.md` for detailed instructions.

