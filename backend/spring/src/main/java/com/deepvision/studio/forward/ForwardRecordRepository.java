package com.deepvision.studio.forward;

import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ForwardRecordRepository extends JpaRepository<ForwardRecord, Long> {
  List<ForwardRecord> findByUserUsernameOrderByCreatedAtDesc(String username);

  Optional<ForwardRecord> findByIdAndUserUsername(Long id, String username);
}

