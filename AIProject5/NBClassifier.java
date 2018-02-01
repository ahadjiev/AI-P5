/**
 * Interface class for the naive bayes classifier.
 * For an explanation of methods, see NBClassifierImpl. 
 * 
 * DO NOT MODIFY
 */

public interface NBClassifier {

	/**
	 * Read training data and learn parameters
	 * 
	 * @param int[][] training data
	 */
	public void fit(int[][] data);

	/**
	 * Classify new dataset
	 * 
	 * @param int[][] test data
	 * @return Label[] classified labels
	 */
	public Label[] classify(int[][] instances);

}