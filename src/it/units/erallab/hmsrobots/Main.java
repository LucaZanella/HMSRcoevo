package it.units.erallab.hmsrobots;

import com.google.common.collect.Lists;
import com.google.common.collect.Range;
import it.units.erallab.hmsrobots.core.controllers.CentralizedMLP;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.*;
import it.units.erallab.hmsrobots.tasks.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.Util;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.CovarianceMatrixAdaptationES;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.StandardEvolver;
import it.units.malelab.jgea.core.evolver.stopcondition.Iterations;
import it.units.malelab.jgea.core.function.Function;
import it.units.malelab.jgea.core.function.NonDeterministicFunction;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.MultiFileListenerFactory;
import it.units.malelab.jgea.core.listener.collector.*;
import it.units.malelab.jgea.core.operator.Crossover;
import it.units.malelab.jgea.core.operator.GeneticOperator;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.ranker.ComparableRanker;
import it.units.malelab.jgea.core.ranker.FitnessComparator;
import it.units.malelab.jgea.core.ranker.ParetoRanker;
import it.units.malelab.jgea.core.ranker.selector.Tournament;
import it.units.malelab.jgea.core.ranker.selector.Worst;
import it.units.malelab.jgea.representation.sequence.Sequence;
import it.units.malelab.jgea.representation.sequence.numeric.DoubleSequenceFactory;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;
import org.apache.commons.lang3.SerializationUtils;

import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.stream.Collectors;

import static it.units.malelab.jgea.core.util.Args.*;

public class Main extends Worker {

    public Main(String[] args) throws FileNotFoundException {
        super(args);
    }

    public static void main(String[] args) throws FileNotFoundException  {
        new Main(args);
    }

    @Override
    public void run() {
        // prepare shapes
        Map<String, Grid<Boolean>> namedShapes = new LinkedHashMap<>();
        namedShapes.put("biped", createShape(new int[]{11, 4}, new int[]{2, 0, 9, 2}));
        // prepare sensor configurations (will be later fixed by removing sensors where no voxels are)
        // centralized mlp
        Map<String, Function<Grid<Boolean>, Grid<SensingVoxel>>> namedSensorConfiguration = new LinkedHashMap<>();
        namedSensorConfiguration.put("blind.0", (Function<Grid<Boolean>, Grid<SensingVoxel>>) (final Grid<Boolean> shape, Listener l) -> Grid.create(shape.getW(), shape.getH(), (Integer x, Integer y) -> {
            List<Sensor> sensors = new ArrayList<>();
            if (y == 0) {
                sensors.add(new Touch());
            }
            if (y == shape.getH() - 1) {
                sensors.add(new Velocity(true, 3d, Velocity.Axis.X, Velocity.Axis.Y));
            }
            sensors.add(new AreaRatio());
            return new SensingVoxel(sensors);
        }));
        namedSensorConfiguration.put("sighted.0", (Function<Grid<Boolean>, Grid<SensingVoxel>>) (final Grid<Boolean> shape, Listener l) -> Grid.create(shape.getW(), shape.getH(), (Integer x, Integer y) -> {
            List<Sensor> sensors = new ArrayList<>();
            if (y == 0) {
                sensors.add(new Touch());
            }
            if (y == shape.getH() - 1) {
                sensors.add(new Velocity(true, 3d, Velocity.Axis.X, Velocity.Axis.Y));
            }
            if (x == shape.getW() - 1) {
                double rayLength = shape.getW() * Voxel.SIDE_LENGTH;
                LinkedHashMap<Lidar.Side, Integer> raysPerSide = new LinkedHashMap<>() {{
                    put(Lidar.Side.E, 5);
                }};
                sensors.add(new Lidar(rayLength, raysPerSide));
            }
            sensors.add(new AreaRatio());
            return new SensingVoxel(sensors);
        }));
        // read parameters
        int[] runs = ri(a("run", "0"));
        List<String> shapeNames = l(a("shape", "biped")); // worm,biped
        List<String> terrainNames = l(a("terrain", "hardcore")); // flat,hardcore
        List<String> evolverNames = l(a("evolver", "standard-2op")); //standard-1|2op,cma-es
        List<String> controllerNames = l(a("controller", "centralizedMLP-0-blind.0"));
        double finalT = d(a("finalT", "60"));
        double minDT = d(a("minDT", "0.0333"));
        double maxDT = d(a("maxDT", "0.0333"));
        List<Double> drivingFrequencies = d(l(a("drivingF", "-1")));
        List<Double> mutationSigmas = d(l(a("mutationSigma", "0.15")));
        int nPop = i(a("npop", "100"));
        int iterations = i(a("iterations", "100"));
        int cacheSize = i(a("cacheSize", "10000"));
        boolean statsToStandardOutput = b(a("stout", "false"));
        List<Locomotion.Metric> metrics = Lists.newArrayList(
                Locomotion.Metric.TRAVEL_X_RELATIVE_VELOCITY
        );
        // prepare things
        MultiFileListenerFactory statsListenerFactory = new MultiFileListenerFactory(a("dir", "."), a("fileStats", null));
        MultiFileListenerFactory serializedBestListenerFactory = new MultiFileListenerFactory(a("dir", "."), a("fileSerialized", null));
        // iterate
        for (int run : runs) {
            for (String shapeName : shapeNames) {
                for (String terrainName : terrainNames) {
                    for (String evolverName : evolverNames) {
                        for (String controllerName : controllerNames) {
                            for (double mutationSigma : mutationSigmas) {
                                for (double drivingFrequency : drivingFrequencies) {
                                    // build problem
                                    LocomotionProblem problem = new LocomotionProblem(
                                            finalT, minDT, maxDT,
                                            Locomotion.createTerrain(terrainName),
                                            metrics,
                                            LocomotionProblem.ApproximationMethod.FINAL_T
                                    );
                                    // prepare robot related things
                                    Grid<Boolean> shape = namedShapes.get(shapeName);
                                    // prepare factory and mapper
                                    Factory<Sequence<Double>> factory = null;
                                    NonDeterministicFunction<Sequence<Double>, Robot> mapper = null;
                                    if (controllerName.startsWith("centralizedMLP") && (shape != null)) {
                                        String sensorConfigurationName = controllerName.split("-")[2];
                                        double innerLayerFactor = Double.parseDouble(controllerName.split("-")[1]);
                                        Grid<SensingVoxel> sensingVoxels = namedSensorConfiguration.get(sensorConfigurationName).apply(shape);
                                        for (Grid.Entry<Boolean> shapeEntry : shape) {
                                            if (!shapeEntry.getValue()) {
                                                sensingVoxels.set(shapeEntry.getX(), shapeEntry.getY(), null);
                                            }
                                        }
                                        int nOfInputs = (int) sensingVoxels.values().stream()
                                                .filter(Objects::nonNull)
                                                .mapToInt(v -> v.getSensors().stream()
                                                        .mapToInt(s -> s.domains().length)
                                                        .sum())
                                                .sum();
                                        int[] innerNeurons;
                                        if (innerLayerFactor == 0d) {
                                            innerNeurons = new int[0];
                                        } else {
                                            innerNeurons = new int[]{(int) Math.round((double) nOfInputs * innerLayerFactor)};
                                        }
                                        Controller controller = new CentralizedMLP(
                                                sensingVoxels,
                                                innerNeurons,
                                                t -> Math.sin(2d * Math.PI * drivingFrequency * t)
                                        );
                                        double[] weights = ((CentralizedMLP) controller).getParams();
                                        int params = weights.length;
                                        factory = new DoubleSequenceFactory(-1d, 1d, params);
                                        mapper = getCentralizedMLPMapper(sensingVoxels, drivingFrequency, innerNeurons);
                                    }
                                    // prepare evolver
                                    Evolver<Sequence<Double>, Robot, List<Double>> evolver = null;
                                    if (evolverName.startsWith("standard")) {
                                        Crossover<Sequence<Double>> crossover = new GeometricCrossover(Range.closedOpen(-1d, 2d));
                                        Mutation<Sequence<Double>> mutation = new GaussianMutation(mutationSigma);
                                        Map<GeneticOperator<Sequence<Double>>, Double> operators = new LinkedHashMap<>();
                                        if (evolverName.split("-")[1].equals("1op")) {
                                            operators.put(crossover.andThen(mutation), 1d);
                                        } else if (evolverName.split("-")[1].equals("2op")) {
                                            operators.put(crossover, 0.8d);
                                            operators.put(mutation, 0.2d);
                                        }
                                        evolver = new StandardEvolver<>(
                                                nPop,
                                                factory,
                                                new ParetoRanker<>(false),
                                                mapper,
                                                operators,
                                                new Tournament<>(Math.max(Math.round(nPop / 30), 2)),
                                                new Worst(),
                                                nPop,
                                                true,
                                                Lists.newArrayList(new Iterations(iterations)),
                                                cacheSize,
                                                false
                                        );
                                    } else if (evolverName.startsWith("cma-es")) {
                                        int size = ((DoubleSequenceFactory) factory).getLength();
                                        double initMin = ((DoubleSequenceFactory) factory).getMin();
                                        double initMax = ((DoubleSequenceFactory) factory).getMax();
                                        evolver = new CovarianceMatrixAdaptationES<>(
                                                size,
                                                initMin,
                                                initMax,
                                                new ParetoRanker<>(false),
                                                mapper,
                                                Lists.newArrayList(new Iterations(iterations)),
                                                cacheSize
                                        );
                                    }
                                    // prepare keys
                                    Map<String, String> keys = new LinkedHashMap<>();
                                    keys.put("evolver", evolverName);
                                    keys.put("controller", controllerName);
                                    keys.put("run", Integer.toString(run));
                                    keys.put("n.pop", Integer.toString(nPop));
                                    keys.put("driving.frequency", Double.toString(drivingFrequency));
                                    keys.put("mutation.sigma", Double.toString(mutationSigma));
                                    keys.put("shape", shapeName);
                                    keys.put("terrain", terrainName);
                                    keys.put("metrics", metrics.stream().map((m) -> m.toString().toLowerCase().replace("_", ".")).collect(Collectors.joining("/")));
                                    L.info(String.format("Keys: %s", keys));
                                    // prepare collectors
                                    List<DataCollector> statsCollectors = Lists.newArrayList(
                                            new Static(keys),
                                            new Basic(),
                                            new Population(),
                                            new Diversity(),
                                            new BestInfo<>(problem.getFitnessFunction(metrics), "%+5.3f"),
                                            new FunctionOfBest<>(
                                                    "valid",
                                                    (Robot robot, Listener listener) -> problem.getTunablePrecisionFitnessFunction(Lists.newArrayList(Locomotion.Metric.values())).apply(SerializationUtils.clone(robot), 0d, new Random(1), listener),
                                                    Arrays.stream(Locomotion.Metric.values()).map((m) -> {
                                                        return m.toString().toLowerCase().replace("_", ".");
                                                    }).collect(Collectors.toList()),
                                                    Collections.singletonList("%+5.3f")
                                            )
                                    );
                                    List<DataCollector> serializedCollectors = Lists.newArrayList(
                                            new Static(keys),
                                            new Basic(),
                                            new BestInfo<>(problem.getFitnessFunction(metrics), "%+5.3f"),
                                            new FunctionOfBest("serialized", (Individual individual) -> Collections.singletonList(new Item("description", Util.lazilySerialize((Serializable) individual.getSolution()), "%s")))
                                    );
                                    // run evolver
                                    Random r = new Random(run);
                                    Listener listener = statsListenerFactory.build(
                                            statsCollectors.toArray(new DataCollector[statsCollectors.size()])
                                    ).then(serializedBestListenerFactory.build(
                                            serializedCollectors.toArracreatey(new DataCollector[serializedCollectors.size()])
                                    ));
                                    if (statsToStandardOutput) {
                                        listener = listener.then(listener(statsCollectors.toArray(new DataCollector[statsCollectors.size()])));
                                    }
                                    try {
                                        evolver.solve(problem, r, executorService, Listener.onExecutor(listener, executorService));
//                                    } catch (InterruptedException | ExecutionException ex) {
//                                        L.log(Level.SEVERE, String.format("Cannot solve problem: %s", ex), ex);
//                                    }
                                    } catch (Exception ex) {
                                        ex.printStackTrace();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private Function<Sequence<Double>, Robot> getCentralizedMLPMapper(Grid<SensingVoxel> sensingVoxels, final double frequency, final int[] innerNeurons) {
        return (Sequence<Double> values, Listener listener) -> {
            double[] weights = new double[values.size()];
            for (int i = 0; i < values.size(); i++) {
                weights[i] = values.get(i);
            }
            Controller controller = new CentralizedMLP(
                    sensingVoxels,
                    innerNeurons,
                    weights,
                    t -> Math.sin(2d * Math.PI * frequency * t)
            );
            return new Robot(controller, SerializationUtils.clone(sensingVoxels));
        };
    }

    private Grid<Boolean> createShape(int[] enclosing, int[]... holes) {
        Grid<Boolean> shape = Grid.create(enclosing[0], enclosing[1], true);
        for (int[] hole : holes) {
            for (int x = hole[0]; x < hole[2]; x++) {
                for (int y = hole[1]; y < hole[3]; y++) {
                    shape.set(x, y, false);
                }
            }
        }
        return shape;
    }

}
