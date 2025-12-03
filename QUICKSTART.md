# 🚀 QUICK START - Progressive Learning System

## What This Does

**Watch your robot learn skills step-by-step!**

1. **Stage 1**: Learn to stand upright (1000 episodes, ~30 min)
2. **Stage 2**: Learn to walk forward (5000 episodes, ~2-3 hours)
3. **Stage 3**: Learn directional walking
4. **Stage 4**: Learn to run
5. **Stage 5**: Navigate obstacles

**Each skill builds on the previous one. All progress is saved!**

---

## Installation (One-Time Setup)

```bash
# Make sure you have the requirements
pip install torch gymnasium mujoco

# That's it!
```

---

## Run Training (Watch It Learn!)

### Option 1: Start From Beginning

```bash
python ProgressiveLearning.py
# Choose: 1 (Start from stage 1)
```

**What you'll see**:
- A window opens showing your humanoid robot
- It will try to stand upright
- Watch as it learns!
- Progress printed every 10 episodes
- Checkpoints saved automatically

### Option 2: Continue Training

```bash
python ProgressiveLearning.py
# Choose: 2 (Continue from checkpoint)
# Enter checkpoint name (e.g., "best_1_stand")
```

### Option 3: Train Specific Stage

```bash
python ProgressiveLearning.py
# Choose: 3 (Train specific stage)
# Enter stage name (e.g., "2_walk_forward")
```

---

## What You'll See

### Terminal Output:
```
Episode 10/1000 | Avg Reward: -25.32 | Avg Length: 12.5 | Best: -30.45
Episode 20/1000 | Avg Reward: -18.45 | Avg Length: 28.3 | Best: -18.45
Episode 30/1000 | Avg Reward: -8.23 | Avg Length: 45.8 | Best: -8.23
...
Episode 500/1000 | Avg Reward: 125.67 | Avg Length: 512.4 | Best: 125.67

✅ STAGE COMPLETED! Achieved 512 steps (target: 500)
💾 Checkpoint saved: checkpoints/completed_1_stand.pt
```

### Visualization Window:
- You'll see the humanoid robot in 3D
- Watch it learn in real-time
- See it fall, get up, try again
- Gradually improve!

---

## Understanding the Training

### Rewards Guide Learning:

**Stage 1 (Standing)**:
- ✅ Reward: Staying upright
- ✅ Reward: Correct height
- ❌ Penalty: Falling
- ❌ Penalty: Too much movement

**Stage 2 (Walking)**:
- ✅ Reward: Moving forward
- ✅ Reward: Staying upright
- ❌ Penalty: Falling
- ❌ Penalty: Wasting energy

The robot learns what gets rewarded!

---

## Checkpoints System

### Saved Checkpoints:
```
checkpoints/
├── best_1_stand.pt              # Best performance on standing
├── completed_1_stand.pt         # Completed standing stage
├── best_2_walk_forward.pt       # Best performance on walking
├── completed_2_walk_forward.pt  # Completed walking stage
└── ...
```

### Each Checkpoint Contains:
- ✅ All learned weights (brain state)
- ✅ Optimizer state
- ✅ Current stage info
- ✅ Episode count
- ✅ Best reward achieved

### Load Any Checkpoint Later:
```python
trainer.load_checkpoint("completed_1_stand")
# Robot now knows how to stand!

trainer.load_checkpoint("completed_2_walk_forward")
# Robot now knows how to stand AND walk!
```

---

## Typical Training Timeline

| Stage | Episodes | Time (GPU) | Time (CPU) | What You'll See |
|-------|----------|------------|------------|-----------------|
| **Standing** | 1,000 | ~20 min | ~45 min | Robot wobbles → balances → stands |
| **Walking** | 5,000 | ~2 hours | ~5 hours | Stumbles → shuffles → walks |
| **Direction** | 10,000 | ~4 hours | ~10 hours | Turns, changes direction |
| **Running** | 15,000 | ~6 hours | ~15 hours | Faster movement, dynamic gait |
| **Obstacles** | 20,000 | ~8 hours | ~20 hours | Avoids, steps over objects |

**Total**: ~20-24 hours on GPU, ~50 hours on CPU

---

## Tips for Success

### 1. **Be Patient**
- First 50-100 episodes: Robot falls a lot (this is normal!)
- Episodes 100-300: Small improvements
- Episodes 300-500: Rapid learning
- Episodes 500+: Refinement

### 2. **Watch the Learning**
- See average length increase = robot staying up longer!
- See reward increase = robot doing better!
- Falls are OK - that's how it learns!

### 3. **Save Often**
- Checkpoints auto-save when improving
- If you stop training, your progress is saved
- Can resume anytime

### 4. **Speed vs Visualization**
- `render=True`: Watch it learn (slower)
- `render=False`: Train faster (no visualization)

```python
trainer = SimpleProgressiveTrainer(
    render=False,  # Change to False for faster training
)
```

---

## Customization

### Change Training Length:
Edit `ProgressiveLearning.py`:
```python
"1_stand": {
    "episodes": 2000,  # Change from 1000 to 2000
    ...
}
```

### Add Your Own Stage:
```python
"6_my_skill": {
    "name": "My Custom Skill",
    "description": "Do something cool",
    "episodes": 5000,
    "success_threshold": 1000,
    "reward_fn": "my_reward_function",
}
```

Then add reward function:
```python
@staticmethod
def my_reward_function(obs, action, next_obs, done, info):
    # Your reward logic here
    return reward
```

---

## Troubleshooting

### "Robot keeps falling immediately"
- ✅ Normal in first 50-100 episodes!
- ✅ Wait for learning to kick in
- ⚠️ If still falling after 500 episodes, increase reward for staying upright

### "Training is too slow"
- Set `render=False`
- Use GPU: `device="cuda"`
- Train overnight

### "Checkpoint not loading"
- Check exact filename (case-sensitive)
- Look in `checkpoints/` directory
- Use `.list_checkpoints()` to see all

### "Out of memory"
- Reduce model size in `BrainConfig`:
  ```python
  BrainConfig(
      d_model=128,  # Smaller (was 256)
      n_layers=2,   # Fewer (was 3)
  )
  ```

---

## Next Steps After Training

### After Stage 1 (Standing):
```python
# Load and test
trainer.load_checkpoint("completed_1_stand")
# Your robot can now stand! Try deploying to real hardware
```

### After Stage 2 (Walking):
```python
# Load and test
trainer.load_checkpoint("completed_2_walk_forward")
# Your robot can walk! Ready for more complex tasks
```

### After All Stages:
```python
# Load final checkpoint
trainer.load_checkpoint("completed_5_obstacles")
# Your robot has full locomotion skills!
# Now add manipulation, language, etc.
```

---

## How This Uses Your SOTA Architecture

```python
# Your diffusion policy: ✅ Used for action generation
# Your VLM backbone: ⚠️ Not used yet (no vision tasks)
# Your multi-modal fusion: ✅ Ready for when you add vision/touch
# Your temporal memory: ✅ Used to remember past observations
# Your action decoder: ✅ Outputs continuous actions
```

As you progress:
1. ✅ **Now**: Using proprio + temporal memory
2. **Later**: Add vision tasks → VLM backbone activates
3. **Later**: Add manipulation → Full multimodal fusion
4. **Later**: Add language → Complete SOTA system

---

## File Structure

```
JackTheWalker/
├── ProgressiveLearning.py      # Main training script (RUN THIS!)
├── JackBrain                    # Your SOTA architecture
├── checkpoints/                 # Saved progress
│   ├── best_1_stand.pt
│   ├── completed_1_stand.pt
│   └── ...
└── QUICKSTART.md               # This file
```

---

## Example Training Session

```bash
$ python ProgressiveLearning.py

🤖 PROGRESSIVE TRAINER INITIALIZED
   Environment: Humanoid-v4
   Observation dim: 376
   Action dim: 17
   Device: cuda
   Render: True
   Ready to learn!

What would you like to do?
1. Start training from beginning (stage 1: Standing)
2. Continue from checkpoint
3. Train specific stage

Enter choice (1-3): 1

🎯 STARTING STAGE: Standing Balance
======================================================================
   Description: Learn to stand upright without falling
   Episodes: 1000
   Success: 500 steps
   Reward function: standing_reward
======================================================================

Episode 10/1000 | Avg Reward: -45.23 | Avg Length: 8.2 | Best: -50.12
Episode 20/1000 | Avg Reward: -32.45 | Avg Length: 15.7 | Best: -32.45
Episode 30/1000 | Avg Reward: -18.92 | Avg Length: 28.4 | Best: -18.92
...
[Watch robot learn in real-time!]
...
Episode 800/1000 | Avg Reward: 156.78 | Avg Length: 524.8 | Best: 156.78

✅ STAGE COMPLETED! Achieved 524 steps (target: 500)
💾 Checkpoint saved: checkpoints/completed_1_stand.pt

➡️  Moving to next stage: 2_walk_forward
```

---

## 🎉 YOU'RE READY!

Just run:
```bash
python ProgressiveLearning.py
```

And watch your robot learn! 🤖

All progress is automatically saved. You can stop/resume anytime.

---

## Questions?

1. **How long does full training take?** ~20-24 hours on GPU
2. **Can I stop and resume?** Yes! All progress saved
3. **Can I skip stages?** Yes, choose option 3
4. **Does it use my SOTA architecture?** Yes! Your brain is the policy
5. **Will this work on real robot?** Training is in sim, then transfer to real hardware

**Just start training and watch it learn!** 🚀
