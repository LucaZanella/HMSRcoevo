package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.Locomotion;
import it.units.malelab.jgea.core.function.Function;
import it.units.malelab.jgea.core.function.NonDeterministicBiFunction;
import it.units.malelab.jgea.core.function.NonDeterministicFunction;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.problem.surrogate.TunablePrecisionProblem;
import org.dyn4j.dynamics.Settings;

import java.util.List;
import java.util.Random;

public class LocomotionProblem implements TunablePrecisionProblem<Robot, List<Double>> {

    public static enum ApproximationMethod {
        FINAL_T, DT
    };

    private final double maxFinalT;
    private final double minDT;
    private final double maxDT;
    private final double[][] groundProfile;
    private final List<Locomotion.Metric> metrics;
    private final ApproximationMethod approximationMethod;

    public LocomotionProblem(double maxFinalT, double minDT, double maxDT, double[][] groundProfile, List<Locomotion.Metric> metrics, ApproximationMethod approximationMethod) {
        this.maxFinalT = maxFinalT;
        this.minDT = minDT;
        this.maxDT = maxDT;
        this.groundProfile = groundProfile;
        this.metrics = metrics;
        this.approximationMethod = approximationMethod;
    }

    @Override
    public NonDeterministicBiFunction<Robot, Double, List<Double>> getTunablePrecisionFitnessFunction() {
        return getTunablePrecisionFitnessFunction(metrics);
    }

    @Override
    public NonDeterministicFunction<Robot, List<Double>> getFitnessFunction() {
        return getFitnessFunction(metrics);
    }

    protected NonDeterministicBiFunction<Robot, Double, List<Double>> getTunablePrecisionFitnessFunction(List<Locomotion.Metric> localMetrics) {
        return (Robot robot, Double p, Random random, Listener listener) -> {
            double dT = minDT;
            double finalT = maxFinalT;
            if (approximationMethod.equals(ApproximationMethod.FINAL_T)) {
                finalT = maxFinalT * (1d - p);
            } else {
                dT = minDT + p * (maxDT - minDT);
            }
            Settings settings = new Settings();
            settings.setStepFrequency(dT);
            Locomotion locomotion = new Locomotion(finalT, groundProfile, localMetrics, settings);
            List<Double> metricValues = locomotion.apply(robot);
            for (int i = 0; i < metricValues.size(); i++) {
                metricValues.set(i, metricValues.get(i) * (localMetrics.get(i).isToMinimize() ? 1d : (-1d)));
            }
            return metricValues;
        };
    }

    public Function<Robot, List<Double>> getFitnessFunction(List<Locomotion.Metric> localMetrics) {
        NonDeterministicBiFunction<Robot, Double, List<Double>> f = getTunablePrecisionFitnessFunction(localMetrics);
        Random random = new Random(1);
        return (Robot robot, Listener listener) -> {
            return f.apply(robot, 0d, random, listener);
        };
    }

}
