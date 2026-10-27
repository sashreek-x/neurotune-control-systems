# NeuroTune — Control Systems Studio

A software-only educational control systems application inspired by the VIT Control Systems Lab manual:
- **Experiment 8 (primary):** PID controller study and automatic tuning.
- **Experiment 7:** time-domain specifications (overshoot, rise time, settling time, steady-state error).
- **Experiment 9:** root-locus visualization and pole stability.
- **Experiment 10:** Bode magnitude/phase plots and approximate gain/phase margins.
- **Experiment 6:** constant-feedback block-diagram reduction.

## Features
- Responsive web dashboard.
- Preset and custom continuous-time transfer functions.
- Open-loop versus PID closed-loop step-response comparison.
- Deterministic numerical search for PID gains with target overshoot/settling-time penalties.
- Closed-loop poles and stability indicator.
- Bode magnitude and phase plots; approximate margins.
- Root-locus proportional-gain sweep.
- Constant-feedback reduction `T(s) = G(s)/(1 + G(s)H(s))`.
- Export latest simulation as JSON or text report.
- No paid API or account required.

## Requirements
Python 3.10+ is recommended. An internet connection is needed for the Plotly chart library loaded from its public CDN. Numerical simulation and API calculations run locally.

## Run on macOS
1. Unzip the project.
2. Open Terminal and change into the project folder:
   ```bash
   cd neurotune
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Start the application:
   ```bash
   python app.py
   ```
6. Open `http://127.0.0.1:5000` in your browser.
7. Stop with `Ctrl+C`.

## Run tests
With the virtual environment active:
```bash
python -m unittest discover -s tests -v
```

## Deployment
A simple container setup is included. Use a Python-capable host or Docker:
```bash
docker build -t neurotune .
docker run --rm -p 5000:5000 neurotune
```
For production deployment, configure the host to bind to `0.0.0.0` and use a production WSGI server such as Gunicorn. The included local app binds to `127.0.0.1` for safe local use.

## Mathematical model
For plant `G(s) = N(s)/D(s)` and ideal PID `C(s) = Kp + Ki/s + Kd*s`, the implementation writes
`C(s) = (Kd*s^2 + Kp*s + Ki)/s` and uses unity negative feedback:
`T(s) = G(s)C(s) / (1 + G(s)C(s))`.

The optimizer uses a deterministic Nelder–Mead search in log-gain space, penalizing unstable poles, excessive overshoot, long settling/rise time, and steady-state error. This is a heuristic search, not a proof of global optimality.

## Important limitations
- This is a student project and not safety-certified engineering software.
- Custom transfer functions must be proper, continuous-time, and supplied as comma-separated coefficients in descending powers of `s`.
- Block reduction currently supports constant feedback gain, not arbitrary dynamic `H(s)`.
- Bode stability margins are approximate and may be unavailable for some plants.
- The optimizer may not find a feasible controller for every plant/target combination; always inspect the response.
- Plotly is loaded from a CDN, so plots require browser network access. The Python calculations do not use external APIs.
- Saved reports are downloaded by the browser; there is no multi-user cloud database in this version.
