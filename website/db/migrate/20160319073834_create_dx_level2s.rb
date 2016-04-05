class CreateDxLevel2s < ActiveRecord::Migration
  def change
    create_table :dx_level2s do |t|
      t.string :name  # eg. Gastrointestinal
      t.references :dx_level1, index: true, foreign_key: true # eg. abdo
    end
  end
end
