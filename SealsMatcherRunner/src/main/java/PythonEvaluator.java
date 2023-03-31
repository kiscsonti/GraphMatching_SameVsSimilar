import de.uni_mannheim.informatik.dws.melt.matching_base.external.seals.MatcherSeals;
import de.uni_mannheim.informatik.dws.melt.matching_data.TestCase;
import de.uni_mannheim.informatik.dws.melt.matching_data.TrackRepository;
import de.uni_mannheim.informatik.dws.melt.matching_eval.ExecutionResultSet;
import de.uni_mannheim.informatik.dws.melt.matching_eval.Executor;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.EvaluatorAlignmentAnalyzer;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.EvaluatorCSV;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.EvaluatorCopyResults;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.explainer.ExplainerResourceProperty;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.explainer.ExplainerResourceType;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.metalevel.ForwardMatcher;
import eu.sealsproject.platform.res.domain.omt.IOntologyMatchingToolBridge;
import org.apache.commons.io.FileUtils;
import org.apache.jena.vocabulary.RDFS;
import org.apache.jena.vocabulary.SKOS;
import org.apache.commons.cli.*;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.util.*;

public class PythonEvaluator {

    public static CommandLine parseArguments(String[] args){

        Options options = new Options();
        Option inputArg = new Option("i", "input", true, "Input zip name");
        Option outputArg = new Option("o", "output", true, "Output folder name");
        options.addOption(inputArg);
        options.addOption(outputArg);
        HelpFormatter formatter = new HelpFormatter();
        CommandLineParser parser = new DefaultParser();
        CommandLine cmd;
        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("User Profile Info", options);
            System.exit(1);
            return null;
        }
        return cmd;
    }

    public static void main(String[] args) {

        CommandLine cmd = parseArguments(args);
        if (cmd.hasOption("input")) {
            System.out.println("Input: " + cmd.getOptionValue("input"));
        }
        if (cmd.hasOption("output")) {
            System.out.println("Output: " + cmd.getOptionValue("output"));
        }


        System.out.println("Starting is ok!");
//        String java8command= "/home/kardip/Downloads/jdk-8u311-linux-x64/jdk1.8.0_311/bin/java";
//        String java8command= "/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java";

        // CASE 2: SEALS Package
        // If you have problems with your java version, have a look at our user guide on how to manually set
        // a path to JAVA 8 for SEALS: https://dwslab.github.io/melt/matcher-packaging/seals#evaluate-and-re-use-a-seals-package-with-melt
        File sealsFile = null;
        if (cmd.hasOption("input")) {
            sealsFile = loadFile(cmd.getOptionValue("input"));
        }else{
            sealsFile = loadFile("pythonCompiler-1.0-seals.zip");
        }
        System.out.println("FILE: " + (sealsFile == null));
        System.out.println("FILE: " + (sealsFile == null));
        System.out.println("FILE: " + (sealsFile == null));
        MatcherSeals sealsMatcher = new MatcherSeals(sealsFile);
//        sealsMatcher.setJavaCommand(java8command);

        Map<String, IOntologyMatchingToolBridge> matchers = new HashMap<>();
        matchers.put("SEALS Matcher", sealsMatcher);

        List<TestCase> selectedTestcases = TrackRepository.Knowledgegraph.V4.getTestCases();

//        List<TestCase> testcases = TrackRepository.Knowledgegraph.V4.getTestCases();
//        List<TestCase> selectedTestcases = new ArrayList<>();
//        for(TestCase tc: testcases){
//            System.out.println(tc.getName());
//            if (tc.getName().equals("memoryalpha-stexpanded")){
//                selectedTestcases.add(tc);
//            }
//        }

        System.out.println("TEST CASES LEN: " + selectedTestcases.size());
        ExecutionResultSet result = Executor.run(selectedTestcases, matchers);
        System.out.println("Matching done!");
        EvaluatorCSV evaluatorCSV = new EvaluatorCSV(result);

        if (cmd.hasOption("output")) {
            evaluatorCSV.writeToDirectory(new File(cmd.getOptionValue("output")));
        }else{
            evaluatorCSV.writeToDirectory();
        }

//        EvaluatorAlignmentAnalyzer evaluatorAlignmentAnalyzer = new EvaluatorAlignmentAnalyzer(result);
//        evaluatorAlignmentAnalyzer.writeToDirectory();
//        EvaluatorCopyResults evaluatorCopyResults = new EvaluatorCopyResults(result);
//        evaluatorCopyResults.writeToDirectory();
    }


    /**
     * Helper function to load files in class path that contain spaces.
     * @param fileName Name of the file.
     * @return File in case of success, else null.
     */
    private static File loadFile(String fileName){
        try {
            return FileUtils.toFile(PythonEvaluator.class.getClassLoader().getResource(fileName).toURI().toURL());
        } catch (URISyntaxException | MalformedURLException exception){
            exception.printStackTrace();
            System.out.println("Could not load file.");
            return null;
        }
    }
}

