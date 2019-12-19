import java.io.*;
import java.util.*;
import java.util.Map.Entry;

public class DecisionTree {
    private int numTrainRecords;
    private int numAttrs;
    private TreeNode root;

    public DecisionTree(List<List<String>> data) {
        numTrainRecords = data.size();
        numAttrs = data.get(0).size() - 1;
        List<List<String>> trainingData = data;
        root = createTree(trainingData);
    }

    private TreeNode createTree(List<List<String>> train) {
        TreeNode r = new TreeNode();
        r.label = "|ROOT|";
        r.data = train;
        recursiveSplit(r);
        return r;
    }

    public class TreeNode{
        public boolean isLeaf;
        public List< TreeNode> childNode;
        public int splitAttributeM;
        public String classValue;
        public List<List<String>> data;
        public String label;

        public TreeNode() {
            splitAttributeM = -99;
        }
    }

    private class DoubleWrap {
        public double d;
        public DoubleWrap(double d) {
            this.d = d;
        }
    }

    public String evaluate(List<String> record) {
        TreeNode evalNode = root;
        while (true) {
            if (evalNode.isLeaf) {
                return evalNode.classValue;
            } else {
                String recordCategory = record.get(evalNode.splitAttributeM);
                for (TreeNode child : evalNode.childNode) {
                    if (recordCategory.equalsIgnoreCase(child.label)) {
                        evalNode = child;
                        break;
                    }
                }
            }
        }
    }

    private void recursiveSplit(TreeNode parent) {
        if (!parent.isLeaf) {
            String classValue = checkIfLeaf(parent.data);
            if (classValue != null) {
                parent.isLeaf = true;
                parent.classValue = classValue;
                return;
            }
            int numRecordsSub = parent.data.size();
            parent.childNode = new ArrayList<>();
            DoubleWrap lowestE = new DoubleWrap(Double.MAX_VALUE);
            for (int m = 0; m < numAttrs; m++) { 
                List<Integer> dataPointToCheck = new ArrayList<>(); 
                for (int n = 1; n < numRecordsSub; n++) {
                    String classA = getClass(parent.data.get(n - 1));
                    String classB = getClass(parent.data.get(n));
                    if (!classA.equalsIgnoreCase(classB)) {
                        dataPointToCheck.add(n);
                    }
                }
                if (dataPointToCheck.isEmpty()) { 
                    parent.isLeaf = true;
                    parent.classValue = getClass(parent.data.get(0));
                    continue;
                }
                for (int k : dataPointToCheck) {
                    Double x = checkPosition(m, k, numRecordsSub, lowestE, parent);
                    if (lowestE.d == 0) {
                        break;
                    }
                }
                if (lowestE.d == 0) {
                    break;
                }
				
            }
            for (TreeNode Child : parent.childNode) {
                if (Child.data.size() == 1) {
                    Child.isLeaf = true;
                    Child.classValue = getClass(Child.data.get(0));
                } else {
                    classValue = checkIfLeaf(Child.data);
                    if (classValue == null) {
                        Child.isLeaf = false;
                        Child.classValue = null;
                    } else {
                        Child.isLeaf = true;
                        Child.classValue = classValue;
                    }
                }
                if (!Child.isLeaf) {
                    recursiveSplit(Child);
                }
            }
			
        }
    }
    private double checkPosition(int m, int n, int numRecordsSub, DoubleWrap lowestE, TreeNode parent) {
        double entropy = 0;
        if (n < 1) {
            return 0;
        }
        if (n > numRecordsSub) {
            return 0;
        }
        List<String> uniqueCategory = new ArrayList<>(); 
        List<String> uniqueClass = new ArrayList<>(); 
        Map<String, Integer> childFreq = new HashMap<>(); 
        for (List< String> s : parent.data) {
            if (!uniqueCategory.contains(s.get(m).trim())) {
                uniqueCategory.add(s.get(m).trim());
                childFreq.put(s.get(m), 0);
            }

            if (!uniqueClass.contains(getClass(s))) {
                uniqueClass.add(getClass(s));
            }
        }

        Map< String, List< List< String>>> ChildDataMap = new HashMap<>();
        for (String s : uniqueCategory) {
            List<List<String>> childData = new ArrayList<>();
            for (List<String> sNew : parent.data) {
                if (s.trim().equalsIgnoreCase(sNew.get(m).trim())) {
                    childData.add(sNew);
                }
            }
            ChildDataMap.put(s, childData);
        }
        for (Entry< String, List< List< String>>> entry : ChildDataMap.entrySet()) {
            entropy += calculateEntropy(getClassProbs(entry.getValue())) * entry.getValue().size();
        }
        entropy = entropy/((double) numRecordsSub);
        if (entropy < lowestE.d) {
            lowestE.d = entropy;
            parent.splitAttributeM = m;
            List< TreeNode> Children = new ArrayList<>();
            for (Entry< String, List<List< String>>> entry : ChildDataMap.entrySet()) {
                TreeNode Child = new TreeNode();
                Child.data = entry.getValue();
                Child.label = entry.getKey();
                Children.add(Child);
            }
            parent.childNode = Children;
        }
        System.out.println(" Entropy : " +entropy);
        return entropy;
    }

    private List<Double> getClassProbs(List<List<String>> record) {
        double N = record.size();
        HashMap<String, Integer> counts = new HashMap<>();
        for (List< String> s : record) {
            String c = getClass(s);
            if (counts.containsKey(c)) {
                counts.put(c, counts.get(c) + 1);
            } else {
                counts.put(c, 1);
            }
        }
        List< Double> probs = new ArrayList<>();
        for (Entry<String, Integer> entry : counts.entrySet()) {
            double prob = entry.getValue() / N;
            probs.add(prob);
        }
        return probs;
    }

    private double calculateEntropy(List< Double> ps) {
        double e = 0;
        for (double p : ps) {
            if (p != 0) {
                e += p * Math.log(p) / Math.log(2);
            }
        }
        return -e;
    }

    public static String getClass(List<String> record) {
        return record.get(record.size() - 1).trim();
    }

    private String checkIfLeaf(List<List<String>> data) {
        
        String classA = getClass(data.get(0));
        for (List<String> record : data) {
            if (!classA.equalsIgnoreCase(getClass(record))) {
                return null;
            }
        }
        return classA;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the Dataset Path : ");
        String path = sc.nextLine();
        List<List<String>> data = new ArrayList<>();
        try {
            String line;
            BufferedReader br = new BufferedReader(new FileReader(path));
            while ((line = br.readLine()) != null) {
                String[] dataPoints = line.split(",");
                List<String> record = new ArrayList<>(Arrays.asList(dataPoints));
                data.add(record);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        char c;
        do{
            sc = new Scanner(System.in);
            System.out.print("\nEnter the query record : ");
            String query = sc.nextLine();
            DecisionTree dt = new DecisionTree(data);
            System.out.println("Class : " + dt.evaluate(Arrays.asList(query.split(","))));
            System.out.print("Continue? (Y/N) ");
            c = sc.next().charAt(0);
        }while(c!='n' && c!='N');
    }
}
