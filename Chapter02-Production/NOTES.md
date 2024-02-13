# Notes

## Chapter 2: Production

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
