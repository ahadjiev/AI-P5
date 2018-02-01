/***************************************************************************************
  CS540 - Section 2
  Homework Assignment 5: Naive Bayes

  NBClassifierImpl.java
  This is the main class that implements functions for Naive Bayes Algorithm!
  ---------
 *Free to modify anything in this file, except the class name 
  	You are required:
  		- To keep the class name as NBClassifierImpl for testing
  		- Not to import any external libraries
  		- Not to include any packages 
 *Notice: To use this file, you should implement 2 methods below.

	@author: TA 
	@date: April 2017
 *****************************************************************************************/

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class NBClassifierImpl implements NBClassifier {

	private int nFeatures;// The number of features including the class 
	private int[] featureSize;//Size of each features; array of first row
	private List<List<Double[]>> logPosProbs;// parameters of Naive Bayes

	private double positiveLabelTotalCount = 0.0;//keeps track of the number of positive label
	private double negativeLabelTotalCount = 0.0;//keeps track of the number of negative label
	//stores the marginal prob in data set
	private Double[] marginalProbability;//[0] = negMargProbility; [1] =  posMargProbability




	/**
	 * Constructs a new classifier without any trained knowledge.
	 */
	public NBClassifierImpl() {

	}

	/**
	 * Construct a new classifier 
	 * 
	 * @param int[] sizes of all attributes
	 */
	public NBClassifierImpl(int[] features) {
		this.nFeatures = features.length;

		// initialize feature size
		this.featureSize = features.clone();

		this.logPosProbs = new ArrayList<List<Double[]>>(this.nFeatures);
	}



	/**
	 * Read training data and learn parameters
	 * 
	 * @param int[][] training data
	 */
	@Override
	public void fit(int[][] data) {

		//calculate the amount of positive and negative labels in the data set 
		calculateTotal(data);
		

		//don't go to last column since that includes the label
		for (int column = 0; column < nFeatures - 1; column++){
			List<Double[]> attributez = new ArrayList<Double[]>();//stores the probability of every attribute in a column
			for (int featureValue = 0; featureValue < featureSize[column]; featureValue++){
				
				//counts when it is the attribute value with a positive class label
				double positiveLabelCounter = 0.0;
				//counts when it is the attribute value with a negative class label
				double negativeLabelCounter = 0.0;
				//row = 1 since we don't use row 0
				for (int row = 0; row < data.length ; row++){
					//class label is last column of data set
					int classLabel = data[row][nFeatures - 1];
					//if value is equal to the attribute value
					if (data[row][column] == featureValue ){
						//if label is positive
						if ( classLabel == 1 )
							positiveLabelCounter++;
						else 
							negativeLabelCounter++;
					}
				}
				//calculate 2 conditional probability for the particular attribute value and store in array
				Double[] valz = getAttrConditionalProb(positiveLabelCounter, negativeLabelCounter, featureSize[column]);
				//add the values pertaining to that particular attribute in a list
				attributez.add(valz);
			}
			logPosProbs.add(attributez);
		}//column for loop
		//calc marginal probability of positive and negative label
		// and stores in the private data field "marginalProbability"
		getMarginalProb(data.length );
		

	}



	public void calculateTotal(int[][] data){
		for (int row = 0; row < data.length; row++){
			int classLabel = data[row][nFeatures - 1];
			if (classLabel == 1)
				positiveLabelTotalCount++;
			else
				negativeLabelTotalCount++;
		}
	}


	/**
	 * Method calculates 2 conditional probabilities for each attribute value and stores in array
	 * @param positiveLabelCounter counts number of occurrences of attributeValue with positive label
	 * @param negativeLabelCounter counts number of occurrences of attributeValue with negative label
	 * @param attributeValue number of values an attribute can hold
	 * @return
	 */
	public Double[] getAttrConditionalProb(double positiveLabelCounter, double negativeLabelCounter, double attributeValue ){
		Double[] probArray = new Double[2];
		//caclulate cond probability for an attr w/ a positive label
		double positiveLabel =  (positiveLabelCounter + 1.0)/(this.positiveLabelTotalCount + attributeValue);
		//caclulate cond probability for an attr w/ a negative label
		double negativeLabel = (negativeLabelCounter + 1.0) /(this.negativeLabelTotalCount + attributeValue);
		probArray[0] = positiveLabel;
		probArray[1] = negativeLabel;
		return probArray;
	}



	/**
	 * Method calc marginal probability of positive and negative label
	 * and stores in the private data field "marginalProbability"
	 * @param numberOfInstances
	 */
	public void getMarginalProb(double numberOfInstances){
		Double[] marginalProbability = new Double[2];
		double posProbability = (positiveLabelTotalCount + 1.0) /(numberOfInstances + 2);
		double negProbability = (negativeLabelTotalCount + 1.0) /(numberOfInstances + 2);
		marginalProbability[0]= negProbability;
		marginalProbability[1]= posProbability;
		this.marginalProbability = marginalProbability;
	}

	/**
	 * Classify new dataset
	 * 
	 * @param int[][] test data
	 * @return Label[] classified labels
	 */
	@Override
	public Label[] classify(int[][] instances) {
		int nrows = instances.length;
		Label[] yPred = new Label[nrows]; // predicted label for each row

		//TO DO
		//stores greatest cumulative probability (+ OR - label) for an instance (row) for each of its attribute
		List<Label> instanceCumProbabilityList = new ArrayList<Label>();
		for (int row = 0; row < nrows ; row++){
		
			double positiveProbability = calcTotalProbability(1.0, instances[row]);
			double negativeProbability =  calcTotalProbability(0.0, instances[row]);
			
			if (positiveProbability >= negativeProbability )
				instanceCumProbabilityList.add(Label.Positive);
			else
				instanceCumProbabilityList.add(Label.Negative);				
		}
		yPred = instanceCumProbabilityList.toArray(yPred);
		return yPred;
	}



	public double calcTotalProbability(double classLabel, int [] instanceRow){
		double totalProb = 0.0;
		//if negative class label
		if (classLabel == 0.0){
			totalProb += Math.log(marginalProbability[0]);
		}
		else{
			totalProb += Math.log(marginalProbability[1]);
		}

	
		for (int attribute = 0; attribute < nFeatures - 1; attribute++){
			int instanceValue = instanceRow[attribute];
			//get the array in logPosProbs corresponding to attribute and value #
			Double[] valueArray = logPosProbs.get(attribute).get(instanceValue);
			if (classLabel == 0.0)
				totalProb += Math.log(valueArray[1]);
			else
				totalProb += Math.log(valueArray[0]);
		}
		return totalProb;
	}
}
