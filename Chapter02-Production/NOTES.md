# Chapter 02: Production

## Homework Assignment 2

### Questionnaire

1. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.  
 - The model was trained on images of bears that are biased to be good examples of bears in the wild. However, in production we would have to handle video (groups of images) of random bears in the wild, where the picture quality and style is dependent on the camera, the environment, and the bear's behavior.
2. Where do text models currently have a major deficiency?
  - NLP models currently can't be relied upon for correct information.
3. What are possible negative societal implications of text generation models?
  - Persuasisve Misinformation on a massive, global scale. 
4. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
  - Human-in-the-loop (checks all preds)
5. What kind of tabular data is deep learning particularly good at?
 - tabular data with a variety of data types.
6. What's a key downside of directly using a deep learning model for recommendation systems?
  - Reccomendation systems are based on a person's preferences, not on a person's needs.
7. What are the steps of the Drivetrain Approach?
  - Define the objective
  - Identify levers
  - Collect Data
  - Build a model & test different levers.
8. How do the steps of the Drivetrain Approach map to a recommendation system?
   - Objective: To increase sales by recommending products customer's want to buy.
  - Levers: Products, Customers, Environment (Time, Location, etc.).
  - Data: Sales, Customer Preferences, Product Preferences, etc.
  - Model: Advesarial Recommender System, or a Hybrid Recommender System.
9.  Create an image recognition model using data you curate, and deploy it on the web.
10. What is `DataLoaders`?
  - A thin _fastai_ class that can contain multiple pytorch `DataLoader` objects. Normally train & validate.
11. What four things do we need to tell fastai to create `DataLoaders`?
  - What kind of data
  - How to get the data
  - How to label the data
  - How to validate the model (split the data)
12. What does the `splitter` parameter to `DataBlock` do?
  - It splits the given data into training/validation datasets.
13. How do we ensure a random split always gives the same validation set?
  - Set a random seed.
14. What letters are often used to signify the independent and dependent variables?
  - x, y
15. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
  - Crop: augments the image by selecting portions of the image to train as a new image.
  - Pad: augments the image by adding blank pixels to the image to train as a new image standard to the batch.
  - Squish Resize: augments the image by applying a global transform to the image to train as a new image. 
16. What is data augmentation? Why is it needed?
  - Data Augmentation is the process of creating new data from existing data given data science techniques.. in computer vision this is often done by cropping, padding, and squishing images. It is needed to increase the amount of data available to train the model. Labelled data is scarce and expensive.
17. What is the difference between `item_tfms` and `batch_tfms`?
  - `item_tfms` are applied to each item in the dataset, while `batch_tfms` are applied to each batch in the dataset.    
18. What is a confusion matrix?
  - A classification metric that can calculate different types of classification accuracies based on recall and precision.
19. What does `export` save?
  - Saves the architecture and parameters of the model as a pickle. 
20. What is it called when we use a model for getting predictions, instead of training?
  - Inference.
21. What are IPython widgets?
  - A python library that allows for javascript components to be used in a jupyter botebook.
22. When might you want to use CPU for deployment? When might GPU be better?
  - CPU: When we dont need to do tasks in parallel. Inference is a 'Queue' based process, perfect for CPU.
  - GPU: When batches of jobs need to be processed in parallel.
23. What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
  - The client needs network connection. Inferencing a model across the web introduces network latency on response times, however assuming the server is specialized for computation, then the latency of the network connection might not even be an issue. Bandwidth, is a concern though.
24.  What are three examples of problems that could occur when rolling out a bear warning system in practice?
  - An _Out-of-Domain_-looking bear is not detected.
  - The specific environment(time, location, etc.) is not accounted for by training.
  - The camera hardware (image quality) is not accounted for by training.
25.  What is "out-of-domain data"?
  - Examples of data that the model was not trained on.
26.  What is "domain shift"?
  - When the data the model see's characteristically changes over time.
27.  What are the three steps in the deployment process?
  - Manual Process
  - Limited Scope
  - Gradual Expansion
  - Full-Scale


### Further Research

1. Consider how the Drivetrain Approach maps to a project or problem you're interested in.
   - Objective: Make as much money by betting on NBA Games.
   - Levers: Teams, Management, Roster, Environment (Time, Location, etc.).
   - Data: Game Stats, Player Stats, Team Stats, etc.
   - Model: Binary Classification Neural Network (Win/Loss).
2. When might it be best to avoid certain types of data augmentation?
   - When the data isn't labelled confidently or is not cleaned to the best of the ability.
3. For a project you're interested in applying deep learning to, consider the thought experiment "What would happen if it went really, really well?"
  - For the NBA betting project, I would be able to make enough money to immediately buy a GPU server farm to use to train other models and mine cryptocurrencies.
4. Start a blog, and write your first blog post. For instance, write about what you think deep learning might be useful for in a domain you're interested in.
  

## Notes

### 2.1: Introduction

- Keeping **Constraints** vs. **Capabilities** In Mind

  - It is important to keep in mind the constraints and capabilities of deep learning.
  - Underestimating the constraints might mean that you fail to consider and react to important issues.
  - Underestimating the capabilities means that you might not even try things that could be very beneficial.

```markdown

We often talk to people who underestimate both the constraints and the capabilities of deep learning. 
Both of these can be problems: underestimating the capabilities means that you might not even try things 
that could be very beneficial, and underestimating the constraints might mean that you fail to consider and react to important issues.


```

- Deployment Process

  - **Manual** Process
    - Run model in parallel with existing system or supervisor
    - Human-in-the-loop (checks all preds)
  - **Limited Scope**
    - Time or geographical constraints on deployment.
    - Careful human supervision.
  - **Gradual Expansion**
    - Increase scope (time/geographic) to gradually expand the deployment.
    - Analyze monitoring systems, make sure that the monitoring set in-place can adequately monitor the model performace in the new scope, checking all known edge cases.
    - Monitor the model's edge case performace.
    - Discover new edge cases, and retrain the model on the new edge cases.
  - **Full-Scale**
    - Automated Process
    - Continuous Integration (CI) and Continuous Deployment (CD)

```markdown
> J: I started a company 20 years ago called _Optimal Decisions_ that used machine learning  and optimization to
 help giant insurance companies set their pricing, impacting tens of  billions of dollars of risks. 
 We used the approaches described here to manage the potential  downsides of something going wrong. 
 Also, before we worked with our clients to put anything in production, we tried to simulate the impact 
 by testing the end-to-end system on their previous year's data. 
 It was always quite a nerve-wracking process, putting these new algorithms into production, 
 but every rollout was successful.
```

- Importance of _End-to-End_ **Iterational Development**

  - I find this is extremly important for the following reasons:
    - It helps to identify the most important parts of the project
    - It helps to identify the most difficult parts of the project
    - It helps to identify the most time-consuming parts of the project
    - It helps to identify the most resource-consuming

```markdown
We also suggest that you iterate from end to end in your project; that is, don't spend months 
fine-tuning your model, or polishing the perfect GUI, or labelling the perfect dataset… 
Instead, complete every step as well as you can in a reasonable amount of time, all the way to the end. 
For instance, if your final goal is an application that runs on a mobile phone, then that should be what 
you have after each iteration. But perhaps in the early iterations you take some shortcuts, for instance 
by doing all of the processing on a remote server, and using a simple responsive web application. 
By completing the project end to end, you will see where the trickiest bits are, and which bits make 
the biggest difference to the final result.
    
```

- Deal with **Out-of-Domain** data in _Production_

```markdown

  - It is important to deal with out-of-domain data in production.
  - This can be planned to some degree, through brainstorming all scenarios, 
  including all concievable edge cases. Then have the model tested on the edge cases. 
  Then calculate the model's edge case performace, and if any new edge cases are discovered, 
  then the model should be retrained on the new edge cases.  
  - This is because the data that the model was trained on might not be the same as the data that the model will be used on.

```
