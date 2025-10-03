import init, {
  WasmDuallyFlatManifold,
  WasmFisherInformationMatrix,
  WasmAlphaConnection,
  InfoGeomUtils
} from '@justinelliottcobb/amari-wasm';

async function runInformationGeometryExample() {
  console.log('📊 Information Geometry Examples');
  console.log('=================================');

  // Initialize the WASM module
  await init();

  // Probability distributions for analysis
  console.log('\n1. Probability Distribution Analysis:');

  const dist1 = [0.5, 0.3, 0.2];
  const dist2 = [0.3, 0.4, 0.3];
  const dist3 = [0.1, 0.2, 0.7];

  console.log(`   Distribution P: [${dist1.join(', ')}]`);
  console.log(`   Distribution Q: [${dist2.join(', ')}]`);
  console.log(`   Distribution R: [${dist3.join(', ')}]`);

  // Information geometry manifold
  const manifold = new WasmDuallyFlatManifold(3, 0.0);

  // Basic entropy calculations
  console.log('\n2. Entropy Analysis:');

  const entropy1 = InfoGeomUtils.entropy(dist1);
  const entropy2 = InfoGeomUtils.entropy(dist2);
  const entropy3 = InfoGeomUtils.entropy(dist3);

  console.log(`   H(P) = ${entropy1.toFixed(4)} bits`);
  console.log(`   H(Q) = ${entropy2.toFixed(4)} bits`);
  console.log(`   H(R) = ${entropy3.toFixed(4)} bits`);

  // Cross-entropy between distributions
  console.log('\n3. Cross-Entropy Analysis:');

  const crossEntropy_PQ = InfoGeomUtils.crossEntropy(dist1, dist2);
  const crossEntropy_PR = InfoGeomUtils.crossEntropy(dist1, dist3);

  console.log(`   H(P,Q) = ${crossEntropy_PQ.toFixed(4)}`);
  console.log(`   H(P,R) = ${crossEntropy_PR.toFixed(4)}`);

  // KL divergence calculations
  console.log('\n4. Kullback-Leibler Divergence:');

  const kl_PQ = manifold.klDivergence(dist1, dist2);
  const kl_QP = manifold.klDivergence(dist2, dist1);
  const kl_PR = manifold.klDivergence(dist1, dist3);

  console.log(`   KL(P||Q) = ${kl_PQ.toFixed(4)}`);
  console.log(`   KL(Q||P) = ${kl_QP.toFixed(4)}`);
  console.log(`   KL(P||R) = ${kl_PR.toFixed(4)}`);
  console.log(`   Note: KL divergence is asymmetric`);

  // Jensen-Shannon divergence (symmetric)
  console.log('\n5. Jensen-Shannon Divergence:');

  const js_PQ = manifold.jsDivergence(dist1, dist2);
  const js_QP = manifold.jsDivergence(dist2, dist1);
  const js_PR = manifold.jsDivergence(dist1, dist3);

  console.log(`   JS(P,Q) = ${js_PQ.toFixed(4)}`);
  console.log(`   JS(Q,P) = ${js_QP.toFixed(4)}`);
  console.log(`   JS(P,R) = ${js_PR.toFixed(4)}`);
  console.log(`   Note: JS divergence is symmetric`);

  // Wasserstein distance
  console.log('\n6. Wasserstein Distance (Earth Mover\'s Distance):');

  const wass_PQ = manifold.wassersteinDistance(dist1, dist2);
  const wass_PR = manifold.wassersteinDistance(dist1, dist3);

  console.log(`   W₁(P,Q) = ${wass_PQ.toFixed(4)}`);
  console.log(`   W₁(P,R) = ${wass_PR.toFixed(4)}`);

  // Fisher Information Matrix analysis
  console.log('\n7. Fisher Information Matrix:');

  const fisherMatrix = manifold.fisherMetricAt(dist1);
  const eigenvalues = fisherMatrix.getEigenvalues();
  const conditionNumber = fisherMatrix.conditionNumber();
  const isPositiveDefinite = fisherMatrix.isPositiveDefinite();

  console.log(`   At point P = [${dist1.join(', ')}]:`);
  console.log(`   Eigenvalues: [${eigenvalues.map(x => x.toFixed(4)).join(', ')}]`);
  console.log(`   Condition number: ${conditionNumber.toFixed(4)}`);
  console.log(`   Positive definite: ${isPositiveDefinite}`);

  // Alpha connections in information geometry
  console.log('\n8. Alpha Connections:');

  const exponentialConn = new WasmAlphaConnection(1.0);
  const mixtureConn = new WasmAlphaConnection(-1.0);
  const leviCivitaConn = new WasmAlphaConnection(0.0);

  console.log(`   Exponential connection (α = 1): ${exponentialConn.isExponential()}`);
  console.log(`   Mixture connection (α = -1): ${mixtureConn.isMixture()}`);
  console.log(`   Levi-Civita connection (α = 0): ${leviCivitaConn.isLeviCivita()}`);

  // Machine learning application: softmax and normalization
  console.log('\n9. Machine Learning Applications:');

  const logits = [2.0, 1.0, 0.1];
  const probabilities = InfoGeomUtils.softmax(logits);
  const normalized = InfoGeomUtils.normalize([3.0, 2.0, 1.0]);

  console.log(`   Logits: [${logits.join(', ')}]`);
  console.log(`   Softmax: [${probabilities.map(x => x.toFixed(4)).join(', ')}]`);
  console.log(`   Raw values: [3, 2, 1]`);
  console.log(`   Normalized: [${normalized.map(x => x.toFixed(4)).join(', ')}]`);

  // Mutual information example
  console.log('\n10. Mutual Information:');

  // Joint distribution P(X,Y) as a 2x3 matrix (flattened)
  const jointDist = [
    0.1, 0.2, 0.1,  // P(X=0, Y=0,1,2)
    0.15, 0.25, 0.2  // P(X=1, Y=0,1,2)
  ];
  const marginalX = [0.4, 0.6];          // P(X=0,1)
  const marginalY = [0.25, 0.45, 0.3];   // P(Y=0,1,2)

  const mutualInfo = InfoGeomUtils.mutualInformation(jointDist, marginalX, marginalY, 2);

  console.log(`   Joint distribution: [${jointDist.map(x => x.toFixed(2)).join(', ')}]`);
  console.log(`   Marginal X: [${marginalX.join(', ')}]`);
  console.log(`   Marginal Y: [${marginalY.join(', ')}]`);
  console.log(`   Mutual Information I(X;Y) = ${mutualInfo.toFixed(4)}`);

  // Statistical distance comparison
  console.log('\n11. Distance Metric Comparison:');

  const testDist1 = [0.6, 0.3, 0.1];
  const testDist2 = [0.4, 0.4, 0.2];

  const klDiv = manifold.klDivergence(testDist1, testDist2);
  const jsDiv = manifold.jsDivergence(testDist1, testDist2);
  const wassDist = manifold.wassersteinDistance(testDist1, testDist2);

  console.log(`   Comparing distributions:`);
  console.log(`   P = [${testDist1.join(', ')}]`);
  console.log(`   Q = [${testDist2.join(', ')}]`);
  console.log(`   KL(P||Q) = ${klDiv.toFixed(4)}`);
  console.log(`   JS(P,Q) = ${jsDiv.toFixed(4)}`);
  console.log(`   W₁(P,Q) = ${wassDist.toFixed(4)}`);

  // Generate test distributions for Monte Carlo analysis
  console.log('\n12. Random Distribution Generation:');

  for (let i = 0; i < 3; i++) {
    const randomDist = InfoGeomUtils.randomSimplex(4);
    const entropy = InfoGeomUtils.entropy(randomDist);
    console.log(`   Random dist ${i + 1}: [${randomDist.map(x => x.toFixed(3)).join(', ')}], H = ${entropy.toFixed(3)}`);
  }

  // Information-theoretic clustering example
  console.log('\n13. Information-Theoretic Clustering:');

  const cluster1Center = [0.7, 0.2, 0.1];
  const cluster2Center = [0.1, 0.8, 0.1];
  const dataPoint = [0.6, 0.3, 0.1];

  const distToCluster1 = manifold.jsDivergence(dataPoint, cluster1Center);
  const distToCluster2 = manifold.jsDivergence(dataPoint, cluster2Center);

  console.log(`   Data point: [${dataPoint.join(', ')}]`);
  console.log(`   Cluster 1 center: [${cluster1Center.join(', ')}]`);
  console.log(`   Cluster 2 center: [${cluster2Center.join(', ')}]`);
  console.log(`   Distance to cluster 1: ${distToCluster1.toFixed(4)}`);
  console.log(`   Distance to cluster 2: ${distToCluster2.toFixed(4)}`);
  console.log(`   Assigned to cluster: ${distToCluster1 < distToCluster2 ? '1' : '2'}`);

  console.log('\n✅ Information geometry example completed!');
}

// Run the example
runInformationGeometryExample().catch(console.error);