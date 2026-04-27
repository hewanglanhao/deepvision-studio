package com.deepvision.studio.common;

import java.nio.file.Path;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {
  private final Path uploadsRoot;
  private final Path datasetsRoot;

  public WebConfig(
      @Value("${deepvision.uploads.root}") String uploadsRoot,
      @Value("${deepvision.datasets.root}") String datasetsRoot
  ) {
    this.uploadsRoot = Path.of(uploadsRoot).toAbsolutePath().normalize();
    this.datasetsRoot = Path.of(datasetsRoot).toAbsolutePath().normalize();
  }

  @Override
  public void addResourceHandlers(ResourceHandlerRegistry registry) {
    registry
        .addResourceHandler("/uploads/**")
        .addResourceLocations(uploadsRoot.toUri().toString() + "/");
    registry
        .addResourceHandler("/datasets/**")
        .addResourceLocations(datasetsRoot.toUri().toString() + "/");
  }
}
