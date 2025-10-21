use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hyperwfc::*;
use rand::{SeedableRng, rngs::SmallRng};

fn benchmark_wave_size<const WAVE_SIZE: usize>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    n: usize,
) {
    let mut tileset = Tileset::<_>::default();
    let sea = tileset.add(1.0);
    let beach = tileset.add(0.5);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis::ALL);
    tileset.connect(sea, beach, &Axis::ALL);
    tileset.connect(beach, beach, &Axis::ALL);
    tileset.connect(beach, grass, &Axis::ALL);
    tileset.connect(grass, grass, &Axis::ALL);
    for _ in 0..(n - 8) {
        tileset.add(0.0);
    }

    let mut rng = SmallRng::from_os_rng();

    for i in [5, 10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new(format!("{}_shannon", n), i), i, |b, &i| {
            b.iter(|| {
                let mut wfc = tileset.create_wfc::<ShannonEntropy>((i, 100, 1));
                wfc.collapse_all(&mut rng)
            })
        });
        group.bench_with_input(BenchmarkId::new(format!("{}_linear", n), i), i, |b, &i| {
            b.iter(|| {
                let mut wfc = tileset.create_wfc::<LinearEntropy>((i, 100, 1));
                wfc.collapse_all(&mut rng)
            })
        });
    }
}

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("wfc");
    benchmark_wave_size::<4>(&mut group, 128);
    benchmark_wave_size::<4>(&mut group, 256);
    benchmark_wave_size::<8>(&mut group, 512);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
