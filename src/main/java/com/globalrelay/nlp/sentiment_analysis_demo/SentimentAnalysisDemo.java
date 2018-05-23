package com.globalrelay.nlp.sentiment_analysis_demo;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextArea;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class SentimentAnalysisDemo extends JPanel {
    /**
     * Sentiment Analysis demo using JPanel for a nice GUI.
     */
    private static final long serialVersionUID = 1542188548843341591L;

    /** Location to save the output data */
    public static final String OUTPUT_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),
            "hce_dl4j_sentiment_demo/");
    public static final int batchSize = 64;
    public static final int truncateReviewsToLength = 256;
    public static SentimentExampleIterator test;
    public static MultiLayerNetwork net;
    public static WordVectors wordVectors;

    private static final String INVALID_INPUT = "Invalid input. Please try again.";
    private static final String NO_TRAINED_MODEL = "Please train the model using the following command before running the demo: 'java -Xms5120m -Xmx6144m -cp sentiment_analysis_demo-0.0.1-SNAPSHOT-bin.jar com.globalrelay.nlp.sentiment_analysis_demo.Word2VecSentimentRNN'";
    private static final String DEMO_HEADER = "Please ensure you run this demo with the enough heap space to load the massive Google News Word2Vec dataset (6 Gb) using the following command:\n'java -Xms5120m -Xmx6144m -cp sentiment_analysis_demo-0.0.1-SNAPSHOT-bin.jar com.globalrelay.nlp.sentiment_analysis_demo.SentimentAnalysisDemo'";

    private static JTextArea input;
    private static JTextArea output;

    private SentimentAnalysisDemo() {
        setLayout(new BorderLayout());

        JPanel northPanel = new JPanel();
        northPanel.setLayout(new FlowLayout());
        JLabel label = new JLabel("Enter text: ");
        northPanel.add(label);

        input = new JTextArea(6, 50);
        northPanel.add(input);
        add(northPanel, BorderLayout.NORTH);

        JPanel centerPanel = new JPanel();
        JButton btn = new JButton("Run Sentiment Analysis");
        btn.addActionListener(new BtnListener());
        centerPanel.add(btn);
        add(centerPanel, BorderLayout.CENTER);

        JPanel southPanel = new JPanel();
        output = new JTextArea("Output will be returned here.", 10, 50);
        southPanel.add(output);
        add(southPanel, BorderLayout.SOUTH);
    }

    public static void main(String[] args) throws IOException {

        // Verify that files exist in correct directory
        if (!trainedModelExists()) {
            throw new RuntimeException(NO_TRAINED_MODEL);
        }

        // Print header
        System.out.println(DEMO_HEADER);

        // Prepare for sentiment analysis
        String googleResPath = OUTPUT_PATH + "google_news/GoogleNews-vectors-negative300.bin";
        String lmrdResPath = OUTPUT_PATH + "large_movie_review_dataset/aclImdb";
        String trainedResPath = OUTPUT_PATH + "trained.zip";
        File trainedResFile = new File(trainedResPath);

        Nd4j.getMemoryManager().setAutoGcWindow(10000); // https://deeplearning4j.org/workspaces
        // Load static Word2Vec model
        System.out.println("Loading static Word2Vec model from " + googleResPath);
        wordVectors = WordVectorSerializer.loadStaticModel(new File(googleResPath));

        // Load the already trained model
        System.out.println("Loading trained model from " + trainedResFile.getAbsolutePath());
        net = ModelSerializer.restoreMultiLayerNetwork(trainedResFile);
        System.out.println("Model loaded from " + trainedResFile.getAbsolutePath());

        // Creating dl4j DataSetIterators for test set
        System.out.println("Creating DataSetIterators for testing from " + lmrdResPath);
        test = new SentimentExampleIterator(lmrdResPath, wordVectors, batchSize, truncateReviewsToLength, false);

        // Create JFrame to accept input
        JFrame frame = new JFrame("Sentiment Analysis Demo");
        frame.add(new SentimentAnalysisDemo());
        frame.setVisible(true);
        frame.setSize(600, 400);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    }

    private static class BtnListener implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {

            String content = input.getText();
            String sentimentOutput;

            // Do sentiment analysis on input text
            try {
                // Convert to INDArray
                INDArray features = test.loadFeaturesFromString(content, truncateReviewsToLength);
                INDArray networkOutput = net.output(features);
                int timeSeriesLength = networkOutput.size(2);
                INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                        NDArrayIndex.point(timeSeriesLength - 1));

                // Get probabilities and overall sentiment
                double probPos = probabilitiesAtLastWord.getDouble(0);
                double probNeg = probabilitiesAtLastWord.getDouble(1);
                String overall = (probPos >= probNeg) ? "Positive" : "Negative";

                sentimentOutput = "Sentiment: " + overall + " [p(positive)=" + probPos + " p(negative)=" + probNeg
                        + "]";

            } catch (Exception ex) {

                sentimentOutput = INVALID_INPUT;

            }

            // Add output to string
            String outputText = content + "\n" + sentimentOutput;

            // Add to output window
            output.setText("");
            output.append(outputText);
            input.setText("");

        }
    }

    private static boolean trainedModelExists() {

        // First Google News vectors
        String googleExtractedPath = OUTPUT_PATH + "google_news/GoogleNews-vectors-negative300.bin";
        File googleExtractedFile = new File(googleExtractedPath);
        if (!googleExtractedFile.exists()) {
            return false;
        }

        // Then the Large Movie Review Dataset
        String lmrdExtractedPath = OUTPUT_PATH + "large_movie_review_dataset/aclImdb";
        File lmrdExtractedFile = new File(lmrdExtractedPath);
        if (!lmrdExtractedFile.exists()) {
            return false;
        }

        // Finally the trained network
        String trainedResPath = OUTPUT_PATH + "trained.zip";
        File trainedResFile = new File(trainedResPath);
        if (!trainedResFile.exists()) {
            return false;
        }

        // Final sanity check to ensure all three exist
        return googleExtractedFile.exists() && lmrdExtractedFile.exists() && trainedResFile.exists();

    }
}
