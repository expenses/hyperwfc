use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hyperwfc::*;
use rand::{SeedableRng, rngs::SmallRng};

fn benchmark_wave_size<Wave: WaveBitmask>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    n: usize,
) {
    let mut tileset = Tileset::<Wave, _>::default();
    let sea = tileset.add(1.0);
    let beach = tileset.add(0.5);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis::ALL);
    tileset.connect(sea, beach, &Axis::ALL);
    tileset.connect(beach, beach, &Axis::ALL);
    tileset.connect(beach, grass, &Axis::ALL);
    tileset.connect(grass, grass, &Axis::ALL);
    for _ in 0..(n - 1) {
        for _ in 0..8 {
            tileset.add(0.0);
        }
    }

    let mut rng = SmallRng::from_os_rng();

    for i in [5, 10, 25, 50, 100, 250, 500, 1000, 2000].iter() {
        group.bench_with_input(
            BenchmarkId::new(format!("{}_shannon", Wave::bits()), i),
            i,
            |b, &i| {
                b.iter(|| {
                    let mut wfc = tileset.create_wfc::<ShannonEntropy>((i, 100, 1));
                    wfc.collapse_all(&mut rng)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new(format!("{}_linear", Wave::bits()), i),
            i,
            |b, &i| {
                b.iter(|| {
                    let mut wfc = tileset.create_wfc::<LinearEntropy>((i, 100, 1));
                    wfc.collapse_all(&mut rng)
                })
            },
        );
    }
}

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("wfc");
    benchmark_wave_size::<u8>(&mut group, 1);
    benchmark_wave_size::<u16>(&mut group, 2);
    benchmark_wave_size::<u32>(&mut group, 4);
    benchmark_wave_size::<u64>(&mut group, 8);
    benchmark_wave_size::<u128>(&mut group, 16);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
