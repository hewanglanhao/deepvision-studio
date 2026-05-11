package com.deepvision.studio.training;

import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TrainingCheckpointRepository extends JpaRepository<TrainingCheckpoint, Long> {
  List<TrainingCheckpoint> findByUserUsernameOrderByCreatedAtDesc(String username);

  List<TrainingCheckpoint> findByUserUsernameAndDatasetIdOrderByCreatedAtDesc(String username, String datasetId);

  Optional<TrainingCheckpoint> findByIdAndUserUsername(Long id, String username);
}
