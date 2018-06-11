package preprocessForInduced;


import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.*;

/** This class splits and tokenizes text in a CSV file. */
public class SplitTokenize {

	/** Usage: java -cp "*" SplitTokenize inputFile outputFile */
	public static void main(String[] args) throws IOException {
		// set up input and output files
		String inputFile = args[0];
		String outputFile = args[1];
		Integer labelIndex = Integer.valueOf(args[2]);
		Integer textIndex = Integer.valueOf(args[3]);
		Integer docIdIndex = Integer.valueOf(args[4]);
		Boolean shiftLabel = Boolean.valueOf(args[5]);
		Boolean skipHeader = Boolean.valueOf(args[6]);
		
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "UTF-8"));
		System.out.println("Indeces: " + " index for label: " + labelIndex + ", index for text: " + textIndex + ", index for doc id: " + docIdIndex);
		System.out.println("CSV options: shift label: " + shiftLabel + ", skip header: " + skipHeader);
		// Create CoreNLP pipeline
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

		// read and process CSV file
		CSVParser parser = new CSVParserBuilder().withEscapeChar('\0').build();
		CSVReader reader;
		if (skipHeader) {
			reader = new CSVReaderBuilder(new InputStreamReader(new FileInputStream(inputFile), 
					StandardCharsets.UTF_8)).withCSVParser(parser).withSkipLines(1).build();
		}
		else {
			reader = new CSVReaderBuilder(new InputStreamReader(new FileInputStream(inputFile), 
					StandardCharsets.UTF_8)).withCSVParser(parser).build();
		}
				
		String [] nextLine;
		int docCounter = 0;
		try {
			while ((nextLine = reader.readNext()) != null) {
				processDocument(nextLine, pipeline, writer, labelIndex, textIndex, docIdIndex, shiftLabel);
				docCounter ++;
			} 
		} catch (Exception e) {
			System.out.println("Caught exception while processing document #" + docCounter + ": ");
			e.printStackTrace();
			throw e;
		}

		reader.close();
		writer.close();
		System.out.println("Processed " + docCounter + " documents.");
	}

	protected static void processDocument(String[] document, StanfordCoreNLP pipeline, BufferedWriter writer, int labelIndex, int textIndex, int docIdIndex, Boolean shiftLabel) throws IOException {
		String columnSeparator = "<split1>";
		String sentSeparator = "<split2>";
		StringBuilder row = new StringBuilder();
		Integer rating;
		try {
			if (shiftLabel) {
				rating = Integer.valueOf(document[labelIndex])-1;
			}
			else {
				rating = Integer.valueOf(document[labelIndex]);
			}
			row.append(rating).append(columnSeparator);
		} catch (Exception e) {
			System.out.println("Caught exception while processing document " + Arrays.deepToString(document) + ": ");
			e.printStackTrace();
			throw e;
		}
		// split and tokenize text
		StringBuilder text = new StringBuilder();
		Annotation annotation = new Annotation(document[textIndex]);
		pipeline.annotate(annotation);
		List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
		if (sentences != null && ! sentences.isEmpty()) {
			for(CoreMap sentence: sentences) {
				List<String> words = new ArrayList<String>();
				for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
					words.add(token.get(TextAnnotation.class));
				}
				text.append(String.join(" ", words)).append(sentSeparator);
			}
		}
		row.append(text);
		// add document id
		Long id = Long.valueOf(document[docIdIndex]);
		row.append(columnSeparator).append(id);
		
		//write to file
		writer.append(row);
		writer.newLine();
		writer.flush();
	}
}