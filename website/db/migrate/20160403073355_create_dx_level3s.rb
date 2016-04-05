class CreateDxLevel3s < ActiveRecord::Migration
  def change
    create_table :dx_level3s do |t|
      t.string :name  # eg. Gastrointestinal
      t.references :dx_level2, index: true, foreign_key: true # eg. abdo
      t.timestamps null: false
    end
  end
end
