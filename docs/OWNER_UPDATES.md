# Updates for the owner

One line per worker run, newest at the bottom, in plain words. The dashboard
shows the latest ones under the Director's morning report. Format:
`- <UTC date> <HH:MM> · <JACK|LAB|MACHINE> · <one or two plain sentences>`

- 2026-11-15 12:58 · JACK · Five of the seven candidate brains have finished the foraging race and none of them has learned to eat yet. Every one outlives a random brain, but none finds food. The last ones are still running.
- 2026-11-15 13:45 · JACK · The last batch of the food-finding race came back. None of the seven learning brains learned to eat: the best managed less than half a meal per life late on, against a target of five. They all live longer than random moves, but not because they eat. Next is a longer run of the best two, which first needs splitting across Kaggle sessions because one run would go past the 12-hour limit.
- 2026-11-15 14:20 · JACK · No learner learned to forage in 15 lives. In fact a random-mover ate as much as any of them, so I set up a 4× longer test to tell slow learning from none, with the random mover as the check. The first two long runs started on Kaggle; the rest queue behind a two-at-a-time limit on this box.
- 2026-11-15 14:45 · JACK · Jack's eating experiment is running on free cloud machines and needs about 10 more hours. The other jobs waiting for a cloud slot couldn't start, because only two may run at once. I tried re-checking an old vision test here instead, but it took longer than this session allows.
- 2026-11-15 15:20 · MACHINE · Kaggle watchers that only wait no longer block other background jobs, so the stalled experiments can start again. I also found the GPU meter was counting the week wrong: this week's free Kaggle GPU time is almost all used (under an hour left until Saturday), not 16 hours, so the big control-path race must wait until next week.
- 2026-11-15 15:30 · JACK · Started two more of the long 'does he learn to eat over 60 lives' runs on Kaggle's free computers: a random-mover baseline that must NOT look like learning, and one more brain. Four now run overnight; six more wait for free slots.
- 2026-11-15 15:40 · JACK · The first learner in the long eating test finished 60 lives on two worlds: on one it found food a bit more often as it went on, on the other it didn't improve at all, so no sign of real learning yet. Two more learners started on Kaggle; four runs remain.
