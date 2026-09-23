# My Reflection

Before this assignment, I was not sure what training steps and learning rate meant. Now I understand that steps tell us how many times the model updates its parameters. The learning rate controls the size of those updates. We started with 10 steps so I could see the process before doing a longer experiment.

I also learned that training changes numbers inside the model. For example, “customer” had token ID 28 in our first model. The ID helped the model find the word’s embedding, which was a list of 64 numbers. The ID stayed the same during training, but the embedding changed. Gradients helped guide the parameter updates.

One result that surprised me was that lower loss did not always mean a better test score. In our 10-step test, the loss went down, but the correct answers went from 9 out of 48 to 8 out of 48. This helped me understand why we need to look at more than one measurement.

After we added grammar and spatial teaching examples, the trained model’s score improved from 21 to 28 out of 48. But it still made mistakes. For example, one test said, “the lamp is above the desk . the desk is”. The correct choice was “below,” but the model chose “inside” from the four options.

The model knew all the words in this question, so missing vocabulary was not the problem here. This helped me understand that knowing the words is not the same as using the relationship correctly. The model did not correctly connect the information when the sentence changed which object it described.

We also tried changing the word order. We compared “today the teacher is” with “the teacher today is.” Four of the five paired responses were the same. In the other pair, the responses were “walking to the bank now .” and “walking to the bank at the hospital .” The second sounded less natural to me. I saw that changing the wording can affect the response, but not every time.

For temperature, we used the same prompt, “the customer.” With the same random seed, temperature 0.3 gave “recommended the package after checking the price .” At 1.2, it gave “was quiet yesterday .” Both sentences made sense. Some other responses stayed the same across temperatures. I learned that temperature changes how words are sampled without changing the trained model. Our small sample did not show that higher temperature always produced more different sentences.

I also became more careful about calling an answer wrong. For example, “a pencil is inside the” continued with “station today .” This was not what I expected, but it was possible. I needed to look at the full sentence, not only whether it matched my expectation.

If I tried another training experiment, I would add more varied spatial examples with more than one sentence. I would check whether the model could connect the information across sentences. I would keep the test questions separate from the teaching text and report the result even if it did not improve.

For me, the goal was not to get the highest score. It was to understand what changed during learning and what the model still could not do. I am still learning the technical details, but these small experiments helped me connect the ideas with actual results.
