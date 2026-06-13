package com.deepvision.studio.training;

import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TrainingDatasetRepository extends JpaRepository<TrainingDataset, String> {
  List<TrainingDataset> findAllByOrderBySourceAscNameAsc();

  List<TrainingDataset> findBySourceOrderByNameAsc(String source);

  List<TrainingDataset> findBySourceAndOwnerUsernameOrderByNameAsc(String source, String ownerUsername);

  List<TrainingDataset> findBySourceOrOwnerUsernameOrderBySourceAscNameAsc(String source, String ownerUsername);
}
