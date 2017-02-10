package com.kylecorry.lann;

import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;

public interface PersistantData {
	public void save(OutputStream os);

	public void save(File file);

	public void save(String filename);

	public void load(InputStream is);

	public void load(File file);

	public void load(String filename);
}
