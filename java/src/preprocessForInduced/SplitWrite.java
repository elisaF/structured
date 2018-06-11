package preprocessForInduced;


import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.*;

/** This class splits text in a CSV file and writes out to a file. */
public class SplitWrite {

	/** Usage: java -cp "*" SplitTokenize inputFile outputFile */
	public static void main(String[] args) throws IOException {
		// set up input and output files
		String inputFile = args[0];
		String outputDir = args[1];
		Integer textIndex = Integer.valueOf(args[2]);
		Integer docIdIndex = Integer.valueOf(args[3]);
		Boolean skipHeader = Boolean.valueOf(args[4]);
		
		System.out.println("Indeces: " + ", index for text: " + textIndex + ", index for doc id: " + docIdIndex);
		System.out.println("CSV options: skip header: " + skipHeader);
		// Create CoreNLP pipeline
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit");
		props.setProperty("ssplit.newlineIsSentenceBreak", "two");
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
				processDocument(nextLine, pipeline, textIndex, docIdIndex, outputDir);
				docCounter ++;
			} 
		} catch (Exception e) {
			System.out.println("Caught exception while processing document #" + docCounter + ": ");
			e.printStackTrace();
			throw e;
		}

		reader.close();
		System.out.println("Processed " + docCounter + " documents.");
	}

	protected static void processDocument(String[] document, StanfordCoreNLP pipeline, int textIndex, int docIdIndex, String outputDir) throws IOException {
		String file_name = "doc_" + Long.valueOf(document[docIdIndex]) + ".txt";
		Path outputDirPath = Paths.get(outputDir);
		Path outputPath = Paths.get(outputDirPath.toString(), file_name);
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputPath.toString()), "UTF-8"));
		
		// split text
		String doc_text = document[textIndex];
		doc_text = doc_text.replace("\n", "\n\n");
		System.out.println("Doc after: " + doc_text);
		String[] paragraphs = doc_text.split("\\r?\\n");
		System.out.println("Num pars: " + paragraphs.length);
		int sent_counter = 0;
		int char_counter = 0;
		int prev_end_offset = -1;
		//for(String paragraph: paragraphs) {
			//if (paragraph.trim().length() > 0) {
				Annotation annotation = new Annotation(doc_text);
				pipeline.annotate(annotation);
				ListIterator<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class).listIterator();
				if (sentences != null){
					while (sentences.hasNext()) {
						CoreMap sentence = sentences.next();
						ListIterator<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class).listIterator();
						sent_counter++;
						while (tokens.hasNext()) {
							CoreLabel token = tokens.next();
							String word = token.word();
							int start_offset = token.beginPosition();
							int end_offset = token.endPosition();
							/*if (prev_end_offset != start_offset) {
								System.out.println("Add padding");
								char_counter++;
							}*/
							//char_counter += (end_offset - start_offset);
							System.out.println("Word: " + word + ", begin offset: " + start_offset + ", end offset: " + end_offset);
							System.out.println("Sentence: " + sent_counter);// + ", char : " + char_counter);
							/*if (!tokens.hasNext() && !sentences.hasNext()) {
								System.out.println("Reached end of parapgraph!");
								char_counter += 2; //count two new lines
							}
							else if (!tokens.hasNext()) {
								char_counter += 1; //count one new line
							}*/
							//prev_end_offset = end_offset;
						/*CoreLabel sent_label = (CoreLabel) sentence_coremap;
						HasOffset ofs = (HasOffset) sent_label;
						int start = ofs.beginPosition();
						int end = ofs.endPosition();
						System.out.println("Start: " + start + ", end: " + end);*/
						}
						writer.append(sentence.get(TextAnnotation.class));
						writer.newLine();
					//}
				//}
				writer.newLine();
			}
		}
		
		writer.close();
	}
}