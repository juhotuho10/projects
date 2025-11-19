##  Brawlhalla ML
---
- Created in August 2023
- Wanted to experiment with creating an ML bot for the game Brawlhalla: https://store.steampowered.com/app/291550/Brawlhalla/

Note: I haven’t trained an agent yet because it requires real-time training, and I don’t have a custom environment. Training would take several weeks, and I don’t have the time right now.
In the demo video (Demo_recording), I'm playing an offline match against a harmless training bot to showcase functionality.

The program takes black-and-white screenshots of the Brawlhalla window, converts them to a 540x960 numpy array, and feeds this into the ML algorithm.
It captures both health bars and uses changes in the bars to generate rewards:

- The more damage dealt, the higher the reward
- Dying: -10 reward
- Making the opponent die: +10 reward