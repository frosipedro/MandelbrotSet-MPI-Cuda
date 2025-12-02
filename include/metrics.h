#ifndef METRICS_H
#define METRICS_H

typedef struct {
    int num_processes;
    int width;
    int height;
    int max_iterations;
    double x_min, x_max;
    double y_min, y_max;
    double total_time;
    double computation_time;
    double communication_time;
    double io_time;
} Metrics;

void print_metrics(Metrics* m);

#endif // METRICS_H