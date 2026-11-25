# Demo video script (about 80 seconds)

**Setup.** Open https://pairslab.streamlit.app in a clean browser window, 1440x900, light theme,
zoom 100%. Open it once beforehand so the app is awake (free hosting sleeps). Close other tabs.
Record with QuickTime (File > New Screen Recording) and speak at a calm pace. Rehearse once.

| Time | On screen | Say |
|---|---|---|
| 0:00-0:10 | Overview page, top of the page | "This is PairsLab, a pairs-trading research platform. I wanted to find out whether pairs trading still works once you test it honestly." |
| 0:10-0:25 | Scroll to the verdict cards and equity curves | "I pick pairs on two years of data, trade them for the next six months, and repeat 18 times, from 2017 to 2025. The best version has a Sharpe ratio of 0.31 after costs. The deflated Sharpe ratio, which charges for the six strategies I tried, is 0.36. A real edge would need about 0.95." |
| 0:25-0:38 | In-sample vs out-of-sample chart | "On the data the pairs were picked from, the Sharpe is about 3.6. Out of sample it is about 0.3. That gap is why I walk forward." |
| 0:38-0:52 | Pair explorer: pick a window and a pair, show z-score and trade markers | "Here is one pair. The z-score of the spread crosses 2, I enter, and I exit when it comes back. Signals fill at the next day's close, so there is no look-ahead." |
| 0:52-1:04 | Costs page: move the slider 5, 10, 20 bps | "Costs decide it. The strategy trades about 29 times its capital a year. At 10 basis points the edge is gone." |
| 1:04-1:14 | Risk page | "I bootstrapped 10,000 alternative histories for VaR, CVaR and drawdown. The Sharpe interval, minus 0.36 to 0.95, includes zero. The bootstrap loop is in C++ and 4.5 times faster than NumPy." |
| 1:14-1:20 | Method page, limitations section | "Limits: survivorship bias, a missing-data filter that uses the full sample, and today's sector labels. Next I am rebuilding the universe point in time." |

## Tips

- Say the one-sentence result first: it does not beat the deflated Sharpe bar. Recruiters
  respect a clear, honest conclusion.
- Do not read numbers off the screen. Pick three to remember: 0.31, 0.36, 29x.
- To make a GIF for the README: crop the video to the Overview page (about 10 s) and export to
  `docs/demo.gif` with `ffmpeg -i demo.mov -vf "fps=10,scale=900:-1" -t 10 docs/demo.gif`,
  then replace the screenshot line at the top of the README.
- Likely interview questions this answers: why walk forward, what is the deflated Sharpe, why
  the one-day delay, what is survivorship bias.
