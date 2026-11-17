# Viva practice for Scout and URLGuard

## Two-person practical

These are responsibilities for preparation, demonstration and maintenance of the AI-assisted projects. Each partner should also be able to explain the handoff to the other.

| Project | Pankaj | Shivam |
| --- | --- | --- |
| Scout | Trace input through the FIFO queue, find duplicate URLs in the linked list, explain pointers and free each node. Demonstrate the question form and its request. | Extract keywords, score a page by hand, trace stable bubble sort and explain the selection cutoff. Demonstrate the answer and source cards. |
| URLGuard | Clean and label the data, derive the 12 features, explain the stratified split and trace the scanner form into the Python API. | Compare the three classifiers, explain scaling and model selection, interpret the confusion matrix and show the prediction result and Model Insights page. |

## Scout · follow one source

**Pankaj, what is actually in the queue?** The first row of the sample has page ID 1 and array index 0. Enqueue writes 0 at the rear, then advances the rear. Dequeue reads 0 at the front, then advances the front. Either operation takes constant time. At most ten rows enter this queue in one run.

**Where do pointers matter?** The head points to the first accepted source. Following each node’s next pointer lets us compare its URL with the incoming URL. Page 4 repeats page 1’s URL. We finish processing that row before allocating a node. With n accepted pages, a search can visit n nodes. Appending also takes O(n) here because there is no tail pointer.

**Shivam, why does page 1 earn 8?** The question supplies three useful words, advantages, solar and energy. All three occur in its title, worth 6 points. Solar and energy occur in its snippet, worth 2 more. Repeating solar in one field gives the same score. Solar inside solarize earns zero.

**Show one sort pass.** The unique scores arrive as 8, 0, 3, 7. Compare 8 with 0 and keep both in place. Swap 0 with 3, then 0 with 7. After that pass the array is 8, 3, 7, 0. The next pass moves 7 ahead of 3. Swapping only on a strictly smaller left score preserves input order for ties. Bubble sort takes O(n²) comparisons and one temporary record for swaps.

**Where does the answer come from?** C prints ID and score pairs. With this sample, page 2 is dropped because its score is zero and three better pages were selected. Fewer than three unique pages means fewer than three outputs. The route matches IDs to the original pages and sends the selected text to Groq. C itself works offline with the sample.

## URLGuard · follow one prediction

**Pankaj, what is one training example?** A row starts as a URL plus a known label. The original PhiUSIIL coding gives legitimate pages 1. The project computes 1 minus that value so phishing becomes the positive class. Conflicting labels for one normalized URL are removed before identical rows are deduplicated.

**What does the classifier see?** For `https://example.com/login?user=1` the ordered row is `[32, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 6]`. The path is /login, six characters including the slash. The digit is 1 and the keyword is login. Training and prediction use the same names in the same order. The API derives these numbers from text without opening the URL.

**Why three splits?** Train on 21,000 rows, compare candidates on 4,500 validation rows, then measure the chosen model on 4,500 held-out rows. Stratification keeps a similar class mix. Exact normalized URL overlap is checked, but related domains can still appear across splits.

**Shivam, how do the models differ?** Logistic Regression learns feature weights and converts their weighted sum into a probability. KNN uses the labels of five nearby training examples. Random Forest combines predictions from 100 decision trees grown with random sampling and feature selection. The first two use a scaler fitted inside the training pipeline. This keeps a large-range feature such as URL length from dominating KNN distance. Tree splits compare one feature with a threshold and can use the original units.

**Why did Random Forest win?** Its validation phishing F1 was 0.992402. KNN scored 0.986877 and Logistic Regression 0.958456. The gap to KNN was 0.005525, above the 0.005 near-tie margin. Inside that margin the code considers phishing recall, then a fixed simplicity order for candidates within 0.01 of the best remaining recall.

**Read the held-out matrix aloud.** Of the phishing URLs, 1,907 were caught and 13 were missed. Of the legitimate URLs, 2,577 were correctly cleared and 3 were flagged. Precision asks how many flags were correct, 1,907 / 1,910. Recall asks how many phishing URLs were caught, 1,907 / 1,920. F1 is their harmonic mean, about 0.9958. Accuracy counts both types of correct answers, 4,484 / 4,500.

**What would you say about a new link?** The API loads the fitted model, orders the features and returns a predicted class plus its phishing probability. A legitimate login page can contain the word login, and a phishing page can use HTTPS. Describe the result as an estimate learned from the dataset.
