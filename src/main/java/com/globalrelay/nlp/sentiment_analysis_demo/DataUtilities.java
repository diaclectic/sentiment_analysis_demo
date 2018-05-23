package com.globalrelay.nlp.sentiment_analysis_demo;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;

/**
 * Common data utility functions.
 */
public class DataUtilities {

  /**
   * Extract a "gz" file into a local folder.
   * @param inputPath Input file path.
   * @param outputPath Output directory path.
   * @throws IOException IO error.
   */
  public static void extractGz(String inputPath, String outputPath) throws IOException {
    if (inputPath == null || outputPath == null)
        return;
    final int bufferSize = 4096;
    if (!outputPath.endsWith("" + File.separatorChar))
      outputPath = outputPath + File.separatorChar;
    
    try (GzipCompressorInputStream gcis = new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(inputPath)))) {
      try (BufferedOutputStream dest = new BufferedOutputStream(new FileOutputStream(outputPath), bufferSize)){
        byte data[] = new byte[bufferSize];
        int count;
        while ((count = gcis.read(data, 0, bufferSize)) != -1) {
          dest.write(data, 0, count);
        }
        dest.close();
      }
    }
  }

  /**
   * Extract a "tar.gz" file into a local folder.
   * @param inputPath Input file path.
   * @param outputPath Output directory path.
   * @throws IOException IO error.
   */
  public static void extractTarGz(String inputPath, String outputPath) throws IOException {
    if (inputPath == null || outputPath == null)
      return;
    final int bufferSize = 4096;
    if (!outputPath.endsWith("" + File.separatorChar))
      outputPath = outputPath + File.separatorChar;
    try (TarArchiveInputStream tais = new TarArchiveInputStream(
        new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(inputPath))))) {
      TarArchiveEntry entry;
      while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
        if (entry.isDirectory()) {
          new File(outputPath + entry.getName()).mkdirs();
        } else {
          int count;
          byte data[] = new byte[bufferSize];
          FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
          BufferedOutputStream dest = new BufferedOutputStream(fos, bufferSize);
          while ((count = tais.read(data, 0, bufferSize)) != -1) {
            dest.write(data, 0, count);
          }
          dest.close();
        }
      }
    }
  }

}
