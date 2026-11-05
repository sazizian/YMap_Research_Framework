import edu.mit.csail.sdg.alloy4.A4Reporter;
import edu.mit.csail.sdg.ast.Module;
import edu.mit.csail.sdg.parser.CompUtil;
import edu.mit.csail.sdg.translator.A4Options;
import edu.mit.csail.sdg.translator.A4Solution;
import edu.mit.csail.sdg.translator.TranslateAlloyToKodkod;

/** Real Alloy API execution; no mock SAT output and no Python fallback. */
public final class AlloyBridge {
  public static void main(String[] args) throws Exception {
    int dimensions = Integer.parseInt(args[1]);
    long limit = Long.parseLong(args[2]);
    A4Reporter reporter = new A4Reporter();
    Module world = CompUtil.parseEverything_fromFile(reporter, null, args[0]);
    A4Options options = new A4Options();
    System.err.println("YMAP_SOLVER\t" + options.solver);
    A4Solution solution = TranslateAlloyToKodkod.execute_command(reporter, world.getAllReachableSigs(), world.getAllCommands().get(0), options);
    long count = 0;
    while (solution.satisfiable() && (limit <= 0 || count < limit)) {
      StringBuilder line = new StringBuilder("YMAP_CHOICES");
      for (int i = 0; i < dimensions; i++) {
        String atom = solution.eval(CompUtil.parseOneExpression_fromString(world, "D" + i + ".choice")).toString();
        atom = atom.replace("{", "").replace("}", "").trim();
        if (atom.contains("/")) atom = atom.substring(atom.lastIndexOf('/') + 1);
        if (atom.contains("$")) atom = atom.substring(0, atom.lastIndexOf('$'));
        line.append('\t').append(atom);
      }
      System.out.println(line);
      System.out.flush();
      count++;
      if (limit > 0 && count >= limit) break;
      solution = solution.next();
    }
  }
}
