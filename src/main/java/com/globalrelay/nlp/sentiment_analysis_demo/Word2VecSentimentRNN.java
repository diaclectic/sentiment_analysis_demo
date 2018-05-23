package com.globalrelay.nlp.sentiment_analysis_demo;

import java.io.File;
import java.io.InputStream;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.joda.time.DateTimeZone;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Training an RNN with the Word2Vec vectors in order to classify text as either positive or negative. Demonstrating
 * model using Hillary Clinton's email dataset.
 *
 * Pretrained vectors for Word2Vec used are from the Google News word vectors (released under Apache License 2.0 and
 * retreived from here: https://code.google.com/p/word2vec/ )
 *
 * Training and test datasets used are from the Stanford large movie review dataset of IMDB reviews (no license provided,
 * but widely used and the only requirement specified is for citing Stanford when publishing papers; retrieved from here:
 * http://ai.stanford.edu/~amaas/data/sentiment/ ). While not terribly similar to email, this should contain more than
 * sufficient data to determine whether particular words tend to be used in positive or negative sentences/documents.
 *
 * Demo of the model is done using Hillary Clinton's email dataset, retrieved from here:
 * https://www.kaggle.com/kaggle/hillary-clinton-emails
 *
 * Data is copied to a temp folder and extracted there. Once model is trained the first time, it also is saved in the
 * temp folder. Subsequent runs will skip extracting the data and training the model if these files are present already.
 *
 * Model is not yet well tuned, but should be sufficient to demonstrate use case.
 *
 */
public class Word2VecSentimentRNN {

	private static Logger log =
		LoggerFactory.getLogger(Word2VecSentimentRNN.class);

	/** Location to save the output data */
	public static final String OUTPUT_PATH = FilenameUtils.concat(
		System.getProperty("java.io.tmpdir"), "hce_dl4j_sentiment_demo/");

    public static void main(String[] args) throws Exception {

        // Verify that files exist
        verifyData();

		String googleResPath =
			OUTPUT_PATH + "google_news/GoogleNews-vectors-negative300.bin";
		String hceResPath =
			OUTPUT_PATH + "hillary_clinton_emails/combined.csv";
		String lmrdResPath =
			OUTPUT_PATH + "large_movie_review_dataset/aclImdb";
		String trainedResPath = OUTPUT_PATH + "trained.zip";

		int batchSize = 64; // Number of examples in each minibatch
		int truncateReviewsToLength = 256; // Truncate reviews with length (#
											// words) greater than this
		int vectorSize = 300; // Size of the word vectors. 300 in the Google
								// News model
		int nEpochs = 1; // Number of epochs (full passes of training data) to
							// train on
		final int seed = 0; // Seed for reproducibility

		Nd4j.getMemoryManager().setAutoGcWindow(10000); // https://deeplearning4j.org/workspaces

		// Load static Word2Vec model
		System.out.println(
			"Loading static Word2Vec model from " + googleResPath);
		WordVectors wordVectors =
			WordVectorSerializer.loadStaticModel(new File(googleResPath));

		// Check if trained model exists, otherwise train one
		File trainedResFile = new File(trainedResPath);
		if (!trainedResFile.exists()) {

			// Specify the network configuration
	        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .seed(seed)
	            .updater(new Adam(2e-2))
	            .l2(1e-5)
	            .weightInit(WeightInit.XAVIER)
	            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
	            .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)   //https://deeplearning4j.org/workspaces
	            .list()
	            .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
	                .activation(Activation.TANH).build())
	            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
	                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
	            .pretrain(false).backprop(true).build();

			// Initialize the network
			System.out.println("Setting up recurrent neural net");
	        MultiLayerNetwork net = new MultiLayerNetwork(conf);
	        net.init();
	        net.setListeners(new ScoreIterationListener(1));

			// Creating dl4j DataSetIterators for train and test sets
			System.out.println(
				"Creating DataSetIterators for train and test sets from " +
					lmrdResPath);
			SentimentExampleIterator train = new SentimentExampleIterator(
				lmrdResPath, wordVectors, batchSize, truncateReviewsToLength, true);
			SentimentExampleIterator test = new SentimentExampleIterator(
				lmrdResPath, wordVectors, batchSize, truncateReviewsToLength,
				false);

			System.out.println(
				"Training network using the large movie review dataset");
			for (int i = 1; i <= nEpochs; i++) {

				System.out.println("Training model (epoch " + i + "/" + nEpochs + ")");
	            net.fit(train);
	            train.reset();

				// Run evaluation on test set. Since this runs on 25k reviews, it
				// can take a while.
				System.out.println(
					"Epoch " + i +
						" complete. Starting evaluation using test set (this can take a while):");
	            Evaluation evaluation = net.evaluate(test);
	            System.out.println(evaluation.stats());

	        }

			// Training done - save the trained model to avoid requiring retraining
			// later on
			System.out.println("Saving training model at " + trainedResPath);
			File locationToSave = new File(trainedResPath);
			boolean allowFutureUpdates = true;
			ModelSerializer.writeModel(net, locationToSave, allowFutureUpdates);

			// Load the model to print params to verify that they were saved
			// correctly
			MultiLayerNetwork restored =
				ModelSerializer.restoreMultiLayerNetwork(locationToSave);
			System.out.println("Model saved");
			System.out.println(
				"Saved and loaded parameters are equal:      " +
					net.params().equals(restored.params()));
			System.out.println(
				"Saved and loaded configurations are equal:  " +
					net.getLayerWiseConfigurations().equals(
						restored.getLayerWiseConfigurations()));

		}

		// Load the already trained model
		System.out.println(
			"Loading trained model from " + trainedResFile.getAbsolutePath());
		MultiLayerNetwork net =
			ModelSerializer.restoreMultiLayerNetwork(trainedResFile);
		System.out.println(
			"Model loaded from " + trainedResFile.getAbsolutePath());
		// System.out.println("Model parameters: " + net.params());
		// System.out.println("Model configuration: " +
		// net.getLayerWiseConfigurations());

		// Creating dl4j DataSetIterators for test set
		System.out.println(
			"Creating DataSetIterators for testing from " + lmrdResPath);
		SentimentExampleIterator test = new SentimentExampleIterator(
			lmrdResPath, wordVectors, batchSize, truncateReviewsToLength,
			false);

		// Load Hillary Clintons email dataset
		System.out.println(
				"Loading Hillary Clintons email dataset using Spark from " + hceResPath);

		// Setup the Spark schema builder
		 Schema inputDataSchema = new Schema.Builder()
		 .addColumnsDouble("Id")
		 .addColumnsString("DocNumber", "MetadataSubject", "MetadataTo",
		 "MetadataFrom")
		 .addColumnTime("MetadataDateSent", DateTimeZone.UTC)
		 .addColumnTime("MetadataDateReleased", DateTimeZone.UTC)
		 .addColumnsString("MetadataPdfLink", "MetadataCaseNumber",
		 "MetadataDocumentClass", "ExtractedSubject", "ExtractedTo",
		 "ExtractedFrom", "ExtractedCc")
		 .addColumnTime("ExtractedDateSent", DateTimeZone.UTC)
		 .addColumnsString("ExtractedCaseNumber", "ExtractedDocNumber")
		 .addColumnTime("ExtractedDateReleased", DateTimeZone.UTC)
		 .addColumnCategorical(
		 "ExtractedReleaseInPartOrFull", "RELEASE IN FULL",
		 "RELEASE IN PART", "UNKNOWN")
		 .addColumnsString("ExtractedBodyText", "RawText")
		 .addColumnsDouble("SenderPersonId")
		 .addColumnsString("SenderName", "SenderAlias")
		 .addColumnsDouble("RecipientPersonId")
		 .addColumnsString("RecipientName", "RecipientAlias")
		 .build();

		 // Transform the Spark schema - here just dropping everything but the MetadataSubject and ExtractedBodyText columns
		 TransformProcess inputTransformProcess = new
		 TransformProcess.Builder(inputDataSchema)
		 .removeColumns(
		 "DocNumber", "MetadataTo",
		 "MetadataFrom", "MetadataDateSent", "MetadataDateReleased",
		 "MetadataPdfLink", "MetadataCaseNumber",
		 "MetadataDocumentClass", "ExtractedSubject", "ExtractedTo",
		 "ExtractedFrom", "ExtractedCc", "ExtractedDateSent",
		 "ExtractedCaseNumber", "ExtractedDocNumber",
		 "ExtractedDateReleased",
		 "RawText", "SenderName", "SenderAlias",
		 "RecipientName", "RecipientAlias")
		 .categoricalToInteger("ExtractedReleaseInPartOrFull")
		 .removeColumns("Id", "ExtractedReleaseInPartOrFull", "SenderPersonId", "RecipientPersonId")
		 .build();

		 // Setup Spark context
		SparkConf sparkConf = new SparkConf();
		sparkConf.setMaster("local[*]");
		sparkConf.setAppName("Hillary Clinton Email Transform");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		// Get data into a Spark RDD and transform that Spark RDD using the transform process
		JavaRDD<String> stringData = sc.textFile(hceResPath);
		// Get RecordReader to parse data
		int numLinesToSkip = 0;
		String delimiter = ",";
		RecordReader rr = new CSVRecordReader(numLinesToSkip, delimiter);
		// Convert to Writable
		JavaRDD<List<Writable>> parsedInputData =
			stringData.map(new StringToWritablesFunction(rr));
		// Run the transform process
		JavaRDD<List<Writable>> processedData = SparkTransformExecutor.execute(
			parsedInputData, inputTransformProcess);

		// Collect data into list
		List<List<Writable>> processedDataOut = processedData.collect();

		// Iterate over list, printing results for each message
        System.out.println("Iterating over emails, getting sentiment for each (1 second delay per message)");
		for (List<Writable> lo : processedDataOut) {

			StringBuilder sb = new StringBuilder();
	        boolean first = true;
		    for (Writable w : lo) {
	            if (!first)
	                sb.append(" ");
		    	String s = w.toString();
		    	sb.append(s);
	            first = false;
		    }
		    String rawEmail = sb.toString();

			INDArray features = test.loadFeaturesFromString(rawEmail, truncateReviewsToLength);
			INDArray networkOutput = net.output(features);
			int timeSeriesLength = networkOutput.size(2);
			INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

			double probPos = probabilitiesAtLastWord.getDouble(0);
			double probNeg = probabilitiesAtLastWord.getDouble(1);
			String overall = (probPos >= probNeg) ? "Positive" : "Negative";

			System.out.println("-------------------------------");
			System.out.println("Raw email: " + rawEmail);
			System.out.println("Sentiment: " + overall + " [p(positive)=" + probPos + " p(negative)=" + probNeg + "]");

            Thread.sleep(1000);

		}

    }

    public static void verifyData() throws Exception {

		// Create output directory if required
		File directory = new File(OUTPUT_PATH);
        if(!directory.exists()) directory.mkdir();

		// Use ClassLoader to get the compressed resources, ensuring they exist
		ClassLoader classLoader = Word2VecSentimentRNN.class.getClassLoader();

		String _googleResPath =
			"google_news/GoogleNews-vectors-negative300.bin.gz";
		String _hceResPath = "hillary_clinton_emails/combined.tar.gz";
		String _lmrdResPath = "large_movie_review_dataset/aclImdb_v1.tar.gz";
		File _googleResFile =
			new File(classLoader.getResource(_googleResPath).getFile());
		File _hceResFile =
			new File(classLoader.getResource(_hceResPath).getFile());
		File _lmrdResFile =
			new File(classLoader.getResource(_lmrdResPath).getFile());

		// Verify that files exist
		if (!_googleResFile.exists()) {
			try {
				// File doesn't exist - try loading input stream from jar
				InputStream _googleResStream = classLoader.getResourceAsStream(_googleResPath);
			} catch (Exception e) {
			    throw new RuntimeException(
				    "googleResPath [" + _googleResPath + "] does not exist");
			}
        }
		if (!_hceResFile.exists()) {
			try {
				// File doesn't exist - try loading input stream from jar
				InputStream _hceResStream = classLoader.getResourceAsStream(_hceResPath);
			} catch (Exception e) {
				throw new RuntimeException(
						"hceResPath [" + _hceResPath + "] does not exist");
			}
        }
		if (!_lmrdResFile.exists()) {
			try {
				// File doesn't exist - try loading input stream from jar
				InputStream _lmrdResStream = classLoader.getResourceAsStream(_lmrdResPath);
			} catch (Exception e) {
				throw new RuntimeException(
						"lmrdResPath [" + _lmrdResPath + "] does not exist");
			}
        }

		// Get paths for copied and extracted files in output directory
		String googleResPath =
			OUTPUT_PATH + "google_news/GoogleNews-vectors-negative300.bin.gz";
		String googleExtractedPath =
			OUTPUT_PATH + "google_news/GoogleNews-vectors-negative300.bin";
		File googleResFile = new File(googleResPath);
		File googleExtractedFile = new File(googleExtractedPath);

		String hceResPath =
			OUTPUT_PATH + "hillary_clinton_emails/combined.tar.gz";
		String hceExtractedPath =
			OUTPUT_PATH + "hillary_clinton_emails/combined.csv";
		File hceResFile = new File(hceResPath);
		File hceExtractedFile = new File(hceExtractedPath);

		String lmrdResPath =
			OUTPUT_PATH + "large_movie_review_dataset/aclImdb_v1.tar.gz";
		String lmrdExtractedPath =
			OUTPUT_PATH + "large_movie_review_dataset/aclImdb";
		File lmrdResFile = new File(lmrdResPath);
		File lmrdExtractedFile = new File(lmrdExtractedPath);

		/** Copy and extract files to output directory if not already present */

		// First Google News vectors
		if (!googleResFile.exists()) {
			System.out.println(
				"Copying resource to output directory (" + googleResPath + ")");
			if (_googleResFile.exists()) {
			    FileUtils.copyFile(_googleResFile, googleResFile);
			} else {
				// File doesn't exist - try loading input stream from jar
				InputStream _googleResStream = classLoader.getResourceAsStream(_googleResPath);
		        FileUtils.copyInputStreamToFile(_googleResStream, googleResFile);
			}

			System.out.println(
				"Resource (.tar.gz file) copied to " +
					googleResFile.getAbsolutePath());
		}
		else {
			System.out.println(
				"Resource (" + googleResPath +
					") already exists in output directory");
		}
		if (!googleExtractedFile.exists()) {
			// Extract tar.gz file to output directory
			System.out.println(
				"Extracting tar.gz file to output directory (" +
					googleExtractedPath +
					")");
			// DataUtilities.extractTarGz(googleResPath, OUTPUT_PATH);
			DataUtilities.extractGz(googleResPath, googleExtractedPath);
			System.out.println(
				"Resource (.tar.gz file) extracted to " + googleExtractedPath);
		}
		else {
			System.out.println(
				"Data (extracted) already exists at " +
					googleExtractedFile.getAbsolutePath());
		}

		// Then Hillary Clinton's emails
		if (!hceResFile.exists()) {
			System.out.println(
				"Copying resource to output directory (" + hceResPath + ")");
			if (_hceResFile.exists()) {
				FileUtils.copyFile(_hceResFile, hceResFile);
			} else {
				// File doesn't exist - try loading input stream from jar
				InputStream _hceResStream = classLoader.getResourceAsStream(_hceResPath);
		        FileUtils.copyInputStreamToFile(_hceResStream, hceResFile);
			}
			System.out.println(
				"Resource (.tar.gz file) copied to " +
					hceResFile.getAbsolutePath());
		}
		else {
			System.out.println(
				"Resource (" + hceResPath +
					") already exists in output directory");
		}
		if (!hceExtractedFile.exists()) {
			// Extract tar.gz file to output directory
			System.out.println(
				"Extracting tar.gz file to output directory (" +
					hceExtractedPath + ")");
			DataUtilities.extractTarGz(
				hceResPath, hceExtractedFile.getParentFile().getAbsolutePath());
			// DataUtilities.extractGz(hceResPath, hceExtractedPath);
			System.out.println(
				"Resource (.tar.gz file) extracted to " + hceExtractedPath);
		}
		else {
			System.out.println(
				"Data (extracted) already exists at " +
					hceExtractedFile.getAbsolutePath());
		}

		// Finally the Large Movie Review Dataset
		if (!lmrdResFile.exists()) {
			System.out.println(
				"Copying resource to output directory (" + lmrdResPath + ")");
			if (_lmrdResFile.exists()) {
				FileUtils.copyFile(_lmrdResFile, lmrdResFile);
			} else {
				// File doesn't exist - try loading input stream from jar
				InputStream _lmrdResStream = classLoader.getResourceAsStream(_lmrdResPath);
		        FileUtils.copyInputStreamToFile(_lmrdResStream, lmrdResFile);
			}
			System.out.println(
				"Resource (.tar.gz file) copied to " +
					lmrdResFile.getAbsolutePath());
		}
		else {
			System.out.println(
				"Resource (" + lmrdResPath +
					") already exists in output directory");
		}
		if (!lmrdExtractedFile.exists()) {
			// Extract tar.gz file to output directory
			System.out.println(
				"Extracting tar.gz file to output directory (" +
					lmrdExtractedPath + ")");
			DataUtilities.extractTarGz(
				lmrdResPath,
				lmrdExtractedFile.getParentFile().getAbsolutePath());
			// DataUtilities.extractGz(lmrdResPath, lmrdExtractedPath);
			System.out.println(
				"Resource (.tar.gz file) extracted to " + lmrdExtractedPath);
		}
		else {
			System.out.println(
				"Data (extracted) already exists at " +
					lmrdExtractedFile.getAbsolutePath());
		}

		// log.info("Output data directory: " + OUTPUT_PATH);
		System.out.println("Output data directory: " + OUTPUT_PATH);

    }


}
