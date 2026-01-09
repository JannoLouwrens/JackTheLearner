Training approach:
  1. Generate random robot states/actions (numbers)
  2. SymbolicCalculator computes EXACT physics (ground truth)
  3. MathReasoner learns to approximate (neural)
  4. Result: Neural network that understands physics!

NO TEXT! NO TOKENIZER! Pure numerical physics learning!
======================================================================

[*] Creating model...

======================================================================
[*] INITIALIZING NEURO-SYMBOLIC MATH REASONER
======================================================================
[*] Symbolic Rule Bank: 100 physics/chemistry laws
[*] Neural Intuition: 6-layer Transformer
[*] Symbolic Reasoning: Differentiable rule application
[*] Physics Predictor: Simulates physical outcomes
======================================================================

[OK] Model created: 21,975,050 parameters


======================================================================
[*] PHYSICS TRAINER INITIALIZED
======================================================================
Device: cpu
Checkpoint dir: checkpoints
Learning rate: 3e-4 → 1e-5 (cosine)
Training: Neural learns from SymbolicCalculator
NO text, NO tokenizer - pure physics!
======================================================================

[*] Starting training...
    Samples: 100,000
    Epochs: 50
    Batch size: 32

[*] Generating physics datasets...

[*] Symbolic Physics & Chemistry Calculator Initialized
   Physics: F=ma, τ=r×F, E=½mv²+mgh, p=mv, pendulum, projectile
   Chemistry: Bond energies, molecular forces, reactions
   Engine: SymPy (exact symbolic computation)

[*] Physics Dataset (Synthetic)
    Samples: 100,000
    State dim: 256
    Action dim: 17
    Ground truth: SymbolicPhysicsCalculator (EXACT!)
    NO text, NO tokenizer - pure numerical physics!

[*] Symbolic Physics & Chemistry Calculator Initialized
   Physics: F=ma, τ=r×F, E=½mv²+mgh, p=mv, pendulum, projectile
   Chemistry: Bond energies, molecular forces, reactions
   Engine: SymPy (exact symbolic computation)

[*] Physics Dataset (Synthetic)
    Samples: 10,000
    State dim: 256
    Action dim: 17
    Ground truth: SymbolicPhysicsCalculator (EXACT!)
    NO text, NO tokenizer - pure numerical physics!

[*] Resuming training from last checkpoint: checkpoints\latest.pt
[LOAD] Checkpoint loaded from checkpoints\latest.pt
[OK] Resuming from epoch 5. Best loss so far: 1255.7204

======================================================================
EPOCH 6/50
======================================================================
Epoch 6:   0%|                                                                                       | 0/3125 [00:00<?, ?it/s]
======================================================================
[TRAINING EXAMPLE - See what model is learning!]
======================================================================

[INPUT STATE]
  Joint angles/velocities (first 5): [ 0.06183656  0.7816038   0.39679152 -0.5883603   0.45170894]
  Action torques (first 5): [  4.596404 -73.22975    5.200038  14.568355   9.623332]

[SYMBOLIC CALCULATOR (CORRECT ANSWER)]
  Next state (first 5): [ 0.05010613  0.7900522   0.40226838 -0.58652174  0.42241704]
  Physics:
    KE:       14.9358 J
    PE:       197.3126 J
    Total E:  212.2485 J
    Momentum: 38.6469 kg⋅m/s
    Force:    73.5579 N
    Friction: 0.6000
    Stiffness:200.0000 GPa

[NEURAL NETWORK (PREDICTION)]
  Next state (first 5): [-0.00878259  0.00050608 -0.00875841  0.00715487 -0.04959571]
  Physics:
    KE:       18.0784 J
    PE:       -4.2775 J
    Total E:  13.4341 J
    Momentum: 38.9081 kg⋅m/s
    Force:    79.1477 N
    Friction: 0.5484
    Stiffness:57.0457 GPa

[ERROR (Neural - Symbolic)]
  State error: 0.423396
  Physics error: 55.248421

[TOP 5 RULES ACTIVATED]
  1. Rule # 28  weight: 0.010
  2. Rule # 27  weight: 0.010
  3. Rule # 20  weight: 0.010
  4. Rule # 88  weight: 0.010
  5. Rule # 40  weight: 0.010
======================================================================

Epoch 6:  16%|▏| 500/3125 [07:16<36:54,  1.19it/s, loss=1704.4994, dynamics=0.2522, physics=17042.9316, rules=49.7, lr=2.05e-0
======================================================================
[TRAINING EXAMPLE - See what model is learning!]
======================================================================

[INPUT STATE]
  Joint angles/velocities (first 5): [ 0.6110099   0.42007333 -0.10430327 -0.5061693  -0.9521831 ]
  Action torques (first 5): [-12.390075   -3.7249851  -1.0260406  37.628887  -29.929611 ]

[SYMBOLIC CALCULATOR (CORRECT ANSWER)]
  Next state (first 5): [ 0.6007874   0.40099987 -0.10945339 -0.5111253  -0.9536731 ]
  Physics:
    KE:       30.9263 J
    PE:       -53.6869 J
    Total E:  -22.7606 J
    Momentum: 55.6114 kg⋅m/s
    Force:    12.9785 N
    Friction: 0.6000
    Stiffness:200.0000 GPa

[NEURAL NETWORK (PREDICTION)]
  Next state (first 5): [ 0.01081884 -0.00929802 -0.00791874  0.00387147  0.00456961]
  Physics:
    KE:       21.3083 J
    PE:       -1.1697 J
    Total E:  19.1710 J
    Momentum: 42.6399 kg⋅m/s
    Force:    83.5758 N
    Friction: 0.7550
    Stiffness:63.4421 GPa

[ERROR (Neural - Symbolic)]
  State error: 0.430049
  Physics error: 32.460457

[TOP 5 RULES ACTIVATED]
  1. Rule # 68  weight: 0.010
  2. Rule # 23  weight: 0.010
  3. Rule # 37  weight: 0.010
  4. Rule # 55  weight: 0.010
  5. Rule # 58  weight: 0.010
======================================================================

Epoch 6:  32%|▎| 1000/3125 [15:53<59:41,  1.69s/it, loss=1453.0032, dynamics=0.2548, physics=14527.9434, rules=49.7, lr=2.26e-
======================================================================
[TRAINING EXAMPLE - See what model is learning!]
======================================================================

[INPUT STATE]
  Joint angles/velocities (first 5): [-0.550326   -0.38060573  0.7796973  -0.04595727  0.3229484 ]
  Action torques (first 5): [-58.711952 -29.443428  33.17654  -42.959633  41.21175 ]

[SYMBOLIC CALCULATOR (CORRECT ANSWER)]
  Next state (first 5): [-0.55171484 -0.37438232  0.76814336 -0.06944206  0.31117103]
  Physics:
    KE:       10.8846 J
    PE:       376.7743 J
    Total E:  387.6589 J
    Momentum: 32.9918 kg⋅m/s
    Force:    73.5846 N
    Friction: 0.8000
    Stiffness:0.0100 GPa

[NEURAL NETWORK (PREDICTION)]
  Next state (first 5): [-0.00344466 -0.01552149  0.00670195 -0.00527774  0.01202278]
  Physics:
    KE:       20.0488 J
    PE:       3.2084 J
    Total E:  23.0604 J
    Momentum: 40.2395 kg⋅m/s
    Force:    80.5868 N
    Friction: 0.5133
    Stiffness:58.6350 GPa

[ERROR (Neural - Symbolic)]
  State error: 0.388128
  Physics error: 82.056122

[TOP 5 RULES ACTIVATED]
  1. Rule # 68  weight: 0.010
  2. Rule # 20  weight: 0.010
  3. Rule # 33  weight: 0.010
  4. Rule # 98  weight: 0.010
  5. Rule # 37  weight: 0.010
======================================================================

Epoch 6:  48%|▍| 1500/3125 [24:37<09:47,  2.76it/s, loss=883.8245, dynamics=0.2501, physics=8836.2041, rules=40.8, lr=2.45e-04
======================================================================
[TRAINING EXAMPLE - See what model is learning!]
======================================================================

[INPUT STATE]
  Joint angles/velocities (first 5): [-0.566672    0.4714642   0.21555798  0.02003112  0.3437154 ]
  Action torques (first 5): [-116.89347   -24.455608   23.174387   24.921377  -27.620775]

[SYMBOLIC CALCULATOR (CORRECT ANSWER)]
  Next state (first 5): [-0.56720656  0.47814286  0.2239239  -0.02672627  0.33393314]
  Physics:
    KE:       7.1799 J
    PE:       109.8347 J
    Total E:  117.0146 J
    Momentum: 26.7954 kg⋅m/s
    Force:    121.6520 N
    Friction: 0.3000
    Stiffness:70.0000 GPa

[NEURAL NETWORK (PREDICTION)]
  Next state (first 5): [-0.01052846  0.00948018  0.00367937 -0.0303422   0.02838271]
  Physics:
    KE:       17.3673 J
    PE:       -0.7143 J
    Total E:  16.5713 J
    Momentum: 36.8477 kg⋅m/s
    Force:    74.8855 N
    Friction: 0.4249
    Stiffness:53.3706 GPa

[ERROR (Neural - Symbolic)]
  State error: 0.386975
  Physics error: 29.501688

[TOP 5 RULES ACTIVATED]
  1. Rule # 46  weight: 0.010
  2. Rule # 68  weight: 0.010
  3. Rule # 31  weight: 0.010
  4. Rule # 72  weight: 0.010
  5. Rule # 58  weight: 0.010
======================================================================

Epoch 6:  64%|▋| 2000/3125 [31:00<21:15,  1.13s/it, loss=978.3496, dynamics=0.2434, physics=9781.5225, rules=50.0, lr=2.61e-04
======================================================================
[TRAINING EXAMPLE - See what model is learning!]
======================================================================

[INPUT STATE]
  Joint angles/velocities (first 5): [-0.390027   -0.37830207 -0.23685248 -0.26046893  0.3257532 ]
  Action torques (first 5): [ 12.820122  28.061172 -18.183628  33.799152  52.342136]

[SYMBOLIC CALCULATOR (CORRECT ANSWER)]
  Next state (first 5): [-0.3951338  -0.3715625  -0.23598896 -0.25534087  0.3369777 ]
  Physics:
    KE:       4.5154 J
    PE:       -115.7526 J
    Total E:  -111.2372 J
    Momentum: 21.2495 kg⋅m/s
    Force:    35.8110 N
    Friction: 0.6000
    Stiffness:200.0000 GPa

[NEURAL NETWORK (PREDICTION)]
  Next state (first 5): [-0.00783847  0.04321824  0.0027705  -0.00020261  0.00139748]
  Physics:
    KE:       16.9509 J
    PE:       3.1922 J
    Total E:  20.6140 J
    Momentum: 37.8517 kg⋅m/s
    Force:    74.2039 N
    Friction: 0.5407
    Stiffness:54.1624 GPa

[ERROR (Neural - Symbolic)]
  State error: 0.382416
  Physics error: 46.419357

[TOP 5 RULES ACTIVATED]
  1. Rule # 53  weight: 0.010
  2. Rule # 98  weight: 0.010
  3. Rule # 68  weight: 0.010
  4. Rule # 33  weight: 0.010
  5. Rule # 62  weight: 0.010
======================================================================

Epoch 6:  80%|▊| 2500/3125 [40:43<23:46,  2.28s/it, loss=1543.6777, dynamics=0.2583, physics=15434.6543, rules=48.2, lr=2.76e- 
======================================================================
[TRAINING EXAMPLE - See what model is learning!]
======================================================================

[INPUT STATE]
  Joint angles/velocities (first 5): [ 0.08147261 -0.25441888 -0.3610452   1.006409    0.4820338 ]
  Action torques (first 5): [17.587666  52.891983  -2.1573324 21.16293   33.33626  ]

[SYMBOLIC CALCULATOR (CORRECT ANSWER)]
  Next state (first 5): [ 0.10174149 -0.24435507 -0.3650709   1.0134441   0.5031906 ]
  Physics:
    KE:       33.0196 J
    PE:       -179.0673 J
    Total E:  -146.0477 J
    Momentum: 57.4627 kg⋅m/s
    Force:    55.7812 N
    Friction: 0.6000
    Stiffness:200.0000 GPa

[NEURAL NETWORK (PREDICTION)]
  Next state (first 5): [-1.1115689e-02 -1.0649510e-02  1.0102312e-02 -4.2382628e-05
  2.2855183e-02]
  Physics:
    KE:       18.2825 J
    PE:       -2.4139 J
    Total E:  16.0290 J
    Momentum: 38.0053 kg⋅m/s
    Force:    74.2793 N
    Friction: 0.4470
    Stiffness:49.9931 GPa

[ERROR (Neural - Symbolic)]
  State error: 0.383775
  Physics error: 54.168591

[TOP 5 RULES ACTIVATED]
  1. Rule # 33  weight: 0.010
  2. Rule # 98  weight: 0.010
  3. Rule # 28  weight: 0.010
  4. Rule # 57  weight: 0.010
  5. Rule # 34  weight: 0.010
======================================================================

Epoch 6:  96%|▉| 3000/3125 [49:22<01:08,  1.83it/s, loss=1501.1824, dynamics=0.2497, physics=15009.7861, rules=46.9, lr=2.87e- 
======================================================================
[TRAINING EXAMPLE - See what model is learning!]
======================================================================

[INPUT STATE]
  Joint angles/velocities (first 5): [ 0.29543528  0.16042675 -0.26696354 -0.9589314  -0.37686247]
  Action torques (first 5): [ 14.908169 108.53186  -84.290726 -26.456182 -32.1902  ]

[SYMBOLIC CALCULATOR (CORRECT ANSWER)]
  Next state (first 5): [ 0.27637592  0.15375775 -0.2477751  -0.9529681  -0.33344972]
  Physics:
    KE:       48.4957 J
    PE:       -121.5337 J
    Total E:  -73.0380 J
    Momentum: 69.6389 kg⋅m/s
    Force:    138.2257 N
    Friction: 0.3000
    Stiffness:70.0000 GPa

[NEURAL NETWORK (PREDICTION)]
  Next state (first 5): [ 0.00564942 -0.00835103  0.01330386 -0.0094715   0.04760009]
  Physics:
    KE:       18.7982 J
    PE:       7.3894 J
    Total E:  25.9471 J
    Momentum: 40.8413 kg⋅m/s
    Force:    83.7209 N
    Friction: 0.4401
    Stiffness:55.1561 GPa

[ERROR (Neural - Symbolic)]
  State error: 0.386961
  Physics error: 35.600704

[TOP 5 RULES ACTIVATED]
  1. Rule # 40  weight: 0.010
  2. Rule # 33  weight: 0.010
  3. Rule # 58  weight: 0.010
  4. Rule # 37  weight: 0.010
  5. Rule # 17  weight: 0.010
======================================================================

Epoch 6: 100%|█| 3125/3125 [50:23<00:00,  1.03it/s, loss=1500.5195, dynamics=0.2567, physics=15003.0879, rules=46.9, lr=2.89e-
Validating: 100%|███████████████████████████████████████████████████████████████████████████| 313/313 [00:46<00:00,  6.77it/s]

[EPOCH 6 SUMMARY]
  Train loss: 1284.9798
  Val loss: 1311.1375
  Dynamics: 0.2503
  Physics: 12847.7548
  Avg rules used: 50.2
[SAVE] Checkpoint: checkpoints\latest.pt

======================================================================
EPOCH 7/50
======================================================================
Epoch 7:   0%|                                                                                       | 0/3125 [00:00<?, ?it/s] 
======================================================================
[TRAINING EXAMPLE - See what model is learning!]
======================================================================

[INPUT STATE]
  Joint angles/velocities (first 5): [-0.36910585  0.6994569   0.60599005 -0.42175478  0.6050081 ]
  Action torques (first 5): [-19.166834 -35.220642 -82.70066   68.45129  -27.566174]

[SYMBOLIC CALCULATOR (CORRECT ANSWER)]
  Next state (first 5): [-0.37769428  0.7112753   0.5992977  -0.4294215   0.59091985]
  Physics:
    KE:       16.1390 J
    PE:       293.9555 J
    Total E:  310.0945 J
    Momentum: 40.1733 kg⋅m/s
    Force:    91.9090 N
    Friction: 0.5000
    Stiffness:2.0000 GPa

[NEURAL NETWORK (PREDICTION)]
  Next state (first 5): [-0.02573854  0.00608904  0.00911913 -0.02501322 -0.01178542]
  Physics:
    KE:       17.1491 J
    PE:       2.7246 J
    Total E:  20.0322 J
    Momentum: 37.0319 kg⋅m/s
    Force:    77.6771 N
    Friction: 0.7186
    Stiffness:55.6539 GPa

[ERROR (Neural - Symbolic)]
  State error: 0.389118
  Physics error: 65.370567

[TOP 5 RULES ACTIVATED]
  1. Rule # 33  weight: 0.010
  2. Rule # 94  weight: 0.010
  3. Rule # 35  weight: 0.010
  4. Rule # 66  weight: 0.010
  5. Rule # 90  weight: 0.010
======================================================================

Epoch 7:  16%|▏| 500/3125 [05:55<20:57,  2.09it/s, loss=850.0958, dynamics=0.2531, physics=8498.8887, rules=43.0, lr=2.96e-04] 
======================================================================
[TRAINING EXAMPLE - See what model is learning!]
======================================================================

[INPUT STATE]
  Joint angles/velocities (first 5): [-0.331198    0.85578674 -0.3468436   0.10314116  0.2222402 ]
  Action torques (first 5): [ 61.78517    -52.975357     0.52430916  93.07875     29.020199  ]

[SYMBOLIC CALCULATOR (CORRECT ANSWER)]
  Next state (first 5): [-0.3286409   0.8598077  -0.3533558   0.12785523  0.20105006]
  Physics:
    KE:       4.0697 J
    PE:       -173.3210 J
    Total E:  -169.2513 J
    Momentum: 20.1736 kg⋅m/s
    Force:    81.3884 N
    Friction: 0.8000
    Stiffness:0.0100 GPa

[NEURAL NETWORK (PREDICTION)]
  Next state (first 5): [ 0.0015035   0.00320348  0.02288507 -0.00495704 -0.011293  ]
  Physics:
    KE:       18.1118 J
    PE:       6.1754 J
    Total E:  24.9559 J
    Momentum: 39.2317 kg⋅m/s
    Force:    80.0250 N
    Friction: 0.5784
    Stiffness:55.9527 GPa

[ERROR (Neural - Symbolic)]
  State error: 0.436065
  Physics error: 46.446365

[TOP 5 RULES ACTIVATED]
  1. Rule # 68  weight: 0.010
  2. Rule # 98  weight: 0.010
  3. Rule # 53  weight: 0.010
  4. Rule # 25  weight: 0.010
  5. Rule # 55  weight: 0.010
======================================================================

Epoch 7:  32%|▎| 1000/3125 [19:52<49:09,  1.39s/it, loss=1037.0857, dynamics=0.2513, physics=10368.8037, rules=46.8, lr=3.00e-
======================================================================
[TRAINING EXAMPLE - See what model is learning!]
======================================================================

[INPUT STATE]
  Joint angles/velocities (first 5): [-0.47122145 -0.383576   -0.6930526   0.07876182 -0.35109112]
  Action torques (first 5): [-40.18212    -9.925688    2.1565318   3.9470906   6.245656 ]

[SYMBOLIC CALCULATOR (CORRECT ANSWER)]
  Next state (first 5): [-0.46996766 -0.39067724 -0.70063955  0.06268898 -0.35506138]
  Physics:
    KE:       6.8476 J
    PE:       -343.6637 J
    Total E:  -336.8161 J
    Momentum: 26.1679 kg⋅m/s
    Force:    41.4460 N
    Friction: 0.5000
    Stiffness:2.0000 GPa

[NEURAL NETWORK (PREDICTION)]
  Next state (first 5): [-0.02203339  0.01048746  0.00690963 -0.00200133  0.00040675]
  Physics:
    KE:       19.0399 J
    PE:       1.0531 J
    Total E:  20.2791 J
    Momentum: 42.0662 kg⋅m/s
    Force:    83.2826 N
    Friction: 0.4416
    Stiffness:57.2409 GPa

[ERROR (Neural - Symbolic)]
  State error: 0.386924
  Physics error: 82.708717

[TOP 5 RULES ACTIVATED]
  1. Rule # 40  weight: 0.010
  2. Rule # 22  weight: 0.010
  3. Rule # 62  weight: 0.010
  4. Rule # 25  weight: 0.010
  5. Rule # 46  weight: 0.010
======================================================================

Epoch 7:  48%|▍| 1500/3125 [29:28<36:12,  1.34s/it, loss=599.2076, dynamics=0.2496, physics=5990.0405, rules=53.4, lr=3.00e-04
======================================================================
[TRAINING EXAMPLE - See what model is learning!]
======================================================================

[INPUT STATE]
  Joint angles/velocities (first 5): [ 0.309022    0.11912957 -0.6652035  -0.31858954 -0.5277536 ]
  Action torques (first 5): [ 15.510851   9.53374   35.93307  -17.593128   6.694735]

[SYMBOLIC CALCULATOR (CORRECT ANSWER)]
  Next state (first 5): [ 0.3027743   0.10865077 -0.6718723  -0.3123852  -0.5239401 ]
  Physics:
    KE:       12.0820 J
    PE:       -329.5534 J
    Total E:  -317.4714 J
    Momentum: 34.7592 kg⋅m/s
    Force:    40.2823 N
    Friction: 0.8000
    Stiffness:0.0100 GPa

[NEURAL NETWORK (PREDICTION)]
  Next state (first 5): [ 0.00809372 -0.00138705  0.01154156  0.00582111  0.00211884]
  Physics:
    KE:       16.9199 J
    PE:       4.0879 J
    Total E:  21.9796 J
    Momentum: 36.7044 kg⋅m/s
    Force:    73.3567 N
    Friction: 0.5179
    Stiffness:52.5944 GPa

[ERROR (Neural - Symbolic)]
  State error: 0.408822
  Physics error: 76.598793

[TOP 5 RULES ACTIVATED]
  1. Rule # 90  weight: 0.010
  2. Rule # 34  weight: 0.010
  3. Rule # 35  weight: 0.010
  4. Rule # 56  weight: 0.010
  5. Rule # 71  weight: 0.010
======================================================================

Epoch 7:  49%|▍| 1533/3125 [30:29<1:06:48,  2.52s/it, loss=1054.7751, dynamics=0.2493, physics=10545.7188, rules=49.0, lr=2.99